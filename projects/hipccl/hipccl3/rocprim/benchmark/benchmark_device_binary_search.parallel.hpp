// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_BENCHMARK_BINARY_SEARCH_PARALLEL_HPP_
#define ROCPRIM_BENCHMARK_BINARY_SEARCH_PARALLEL_HPP_

#include "benchmark_utils.hpp"

#include "../common/utils_device_ptr.hpp"

#include <rocprim/device/config_types.hpp>
#include <rocprim/device/detail/device_config_helper.hpp>
#include <rocprim/device/device_binary_search.hpp>
#include <rocprim/functional.hpp>

#include <benchmark/benchmark.h>

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
std::string binary_search_config_name()
{
    return "{bs:" + std::to_string(Config::block_size)
           + ",ipt:" + std::to_string(Config::items_per_thread) + "}";
}

template<>
inline std::string binary_search_config_name<rocprim::default_config>()
{
    return "default_config";
}

template<typename SubAlgorithm,
         typename T,
         typename OutputType,
         size_t K,
         bool   SortedNeedles,
         typename Config = rocprim::default_config>
struct device_binary_search_benchmark : public benchmark_utils::autotune_interface
{
    std::string name() const override
    {
        return bench_naming::format_name("{lvl:device,algo:" + SubAlgorithm{}.name()
                                         + ",value_type:" + std::string(Traits<T>::name())
                                         + ",output_type:" + std::string(Traits<OutputType>::name())
                                         + ",cfg:" + binary_search_config_name<Config>() + "}");
    }

    void run(benchmark_utils::state&& state) override
    {
        const auto& bytes  = state.bytes;
        const auto& seed   = state.seed;
        const auto& stream = state.stream;

        size_t needles_bytes = bytes * K / 100;

        using compare_op_type = typename std::
            conditional<std::is_same<T, rocprim::half>::value, half_less, rocprim::less<T>>::type;

        // Calculate the number of elements from byte size
        size_t haystack_size = bytes / sizeof(T);
        size_t needles_size  = needles_bytes / sizeof(T);

        compare_op_type compare_op;

        // Generate data
        std::vector<T> haystack(haystack_size);
        std::iota(haystack.begin(), haystack.end(), 0);

        const auto random_range = limit_random_range<T>(0, haystack_size);

        std::vector<T> needles = get_random_data<T>(needles_size,
                                                    random_range.first,
                                                    random_range.second,
                                                    seed.get_0());
        if(SortedNeedles)
        {
            std::sort(needles.begin(), needles.end(), compare_op);
        }

        common::device_ptr<T>          d_haystack(haystack);
        common::device_ptr<T>          d_needles(needles);
        common::device_ptr<OutputType> d_output(needles_size);

        size_t temporary_storage_bytes;
        auto   dispatch_helper = dispatch_binary_search_helper<Config>();
        HIP_CHECK(dispatch_helper.dispatch_binary_search(SubAlgorithm{},
                                                         nullptr,
                                                         temporary_storage_bytes,
                                                         d_haystack.get(),
                                                         d_needles.get(),
                                                         d_output.get(),
                                                         haystack_size,
                                                         needles_size,
                                                         compare_op,
                                                         stream));

        common::device_ptr<void> d_temporary_storage(temporary_storage_bytes);

        state.run(
            [&]
            {
                HIP_CHECK(dispatch_helper.dispatch_binary_search(SubAlgorithm{},
                                                                 d_temporary_storage.get(),
                                                                 temporary_storage_bytes,
                                                                 d_haystack.get(),
                                                                 d_needles.get(),
                                                                 d_output.get(),
                                                                 haystack_size,
                                                                 needles_size,
                                                                 compare_op,
                                                                 stream));
            });

        state.set_throughput(needles_size, sizeof(T));
    }
};

#endif // ROCPRIM_BENCHMARK_BINARY_SEARCH_PARALLEL_HPP_
