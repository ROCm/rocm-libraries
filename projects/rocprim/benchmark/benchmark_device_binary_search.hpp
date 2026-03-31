// MIT License
//
// Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <rocprim/device/config_types.hpp>
#include <rocprim/device/detail/device_config_helper.hpp>
#include <rocprim/device/device_binary_search.hpp>
#include <rocprim/functional.hpp>

#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

struct binary_search_subalgorithm
{
    std::string name() const
    {
        return "binary_search";
    }
};

struct lower_bound_subalgorithm
{
    std::string name() const
    {
        return "lower_bound";
    }
};

struct upper_bound_subalgorithm
{
    std::string name() const
    {
        return "upper_bound";
    }
};

template<typename Config = rocprim::default_config>
struct dispatch_binary_search_helper
{
    template<typename... Args>
    hipError_t dispatch_binary_search(binary_search_subalgorithm, Args&&... args)
    {
        using config = rocprim::binary_search_config<Config::block_size, Config::items_per_thread>;
        return rocprim::binary_search<config>(std::forward<Args>(args)...);
    }

    template<typename... Args>
    hipError_t dispatch_binary_search(upper_bound_subalgorithm, Args&&... args)
    {
        using config = rocprim::upper_bound_config<Config::block_size, Config::items_per_thread>;
        return rocprim::upper_bound<config>(std::forward<Args>(args)...);
    }

    template<typename... Args>
    hipError_t dispatch_binary_search(lower_bound_subalgorithm, Args&&... args)
    {
        using config = rocprim::lower_bound_config<Config::block_size, Config::items_per_thread>;
        return rocprim::lower_bound<config>(std::forward<Args>(args)...);
    }
};

template<>
struct dispatch_binary_search_helper<rocprim::default_config>
{
    template<typename... Args>
    hipError_t dispatch_binary_search(binary_search_subalgorithm, Args&&... args)
    {
        return rocprim::binary_search<rocprim::default_config>(std::forward<Args>(args)...);
    }

    template<typename... Args>
    hipError_t dispatch_binary_search(upper_bound_subalgorithm, Args&&... args)
    {
        return rocprim::upper_bound<rocprim::default_config>(std::forward<Args>(args)...);
    }

    template<typename... Args>
    hipError_t dispatch_binary_search(lower_bound_subalgorithm, Args&&... args)
    {
        return rocprim::lower_bound<rocprim::default_config>(std::forward<Args>(args)...);
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
        return primbench::json{}.add("bs", config.block_size).add("ipt", config.items_per_thread);
    }
}

template<typename SubAlgorithm,
         typename T,
         typename OutputType,
         size_t K,
         bool   SortedNeedles,
         typename Config = rocprim::default_config>
struct device_binary_search_benchmark : public primbench::benchmark_interface
{
    primbench::json meta() const override
    {
        return primbench::json{}
            .add("lvl", "device")
            .add("algo", "device_binary_search")
            .add("subalgo", SubAlgorithm{}.name())
            .add("key_type", primbench::name<T>())
            .add("output_type", primbench::name<OutputType>())
            .add("needles_percent", K)
            .add("sorted_needles", SortedNeedles)
            .add("cfg", config_name<Config>());
    }

    void run(primbench::state& state) override
    {
        const auto& bytes  = state.size;
        const auto& seed   = state.seed;
        const auto& stream = state.stream;

        size_t needles_bytes = bytes * K / 100;

        using compare_op_type = typename std::
            conditional<std::is_same<T, rocprim::half>::value, half_less, rocprim::less<T>>::type;

        size_t haystack_items = bytes / sizeof(T);
        size_t needles_items  = needles_bytes / sizeof(T);

        compare_op_type compare_op;

        // Generate data
        std::vector<T> haystack(haystack_items);
        std::iota(haystack.begin(), haystack.end(), 0);

        const auto random_range = limit_random_range<T>(0, haystack_items);

        std::vector<T> needles
            = get_random_data<T>(needles_items, random_range.first, random_range.second, seed);
        if(SortedNeedles)
        {
            std::sort(needles.begin(), needles.end(), compare_op);
        }

        common::device_ptr<T>          d_haystack(haystack);
        common::device_ptr<T>          d_needles(needles);
        common::device_ptr<OutputType> d_output(needles_items);

        size_t temporary_storage_bytes;
        auto   dispatch_helper = dispatch_binary_search_helper<Config>();
        HIP_CHECK(dispatch_helper.dispatch_binary_search(SubAlgorithm{},
                                                         nullptr,
                                                         temporary_storage_bytes,
                                                         d_haystack.get(),
                                                         d_needles.get(),
                                                         d_output.get(),
                                                         haystack_items,
                                                         needles_items,
                                                         compare_op,
                                                         stream));

        common::device_ptr<void> d_temporary_storage(temporary_storage_bytes);

        state.set_items(needles_items);
        state.add_reads<T>(needles_items);

        state.run(
            [&]
            {
                HIP_CHECK(dispatch_helper.dispatch_binary_search(SubAlgorithm{},
                                                                 d_temporary_storage.get(),
                                                                 temporary_storage_bytes,
                                                                 d_haystack.get(),
                                                                 d_needles.get(),
                                                                 d_output.get(),
                                                                 haystack_items,
                                                                 needles_items,
                                                                 compare_op,
                                                                 stream));
            });
    }
};
