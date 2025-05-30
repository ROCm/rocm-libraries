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

#ifndef ROCPRIM_BENCHMARK_DEVICE_TRANSFORM_PARALLEL_HPP_
#define ROCPRIM_BENCHMARK_DEVICE_TRANSFORM_PARALLEL_HPP_

#include "benchmark_utils.hpp"

#include "../common/utils_device_ptr.hpp"

// Google Benchmark
#include <benchmark/benchmark.h>

// HIP API
#include <hip/hip_runtime_api.h>

// rocPRIM
#include <rocprim/device/config_types.hpp>
#include <rocprim/device/detail/device_config_helper.hpp>
#include <rocprim/device/device_transform.hpp>
#include <rocprim/functional.hpp>

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

template<typename Config>
std::string transform_config_name()
{
    auto config = Config();
    return "{bs:" + std::to_string(config.block_size)
           + ",ipt:" + std::to_string(config.items_per_thread)
           + ",lt:" + get_thread_load_method_name(config.load_type) + "}";
}

template<>
inline std::string transform_config_name<rocprim::default_config>()
{
    return "default_config";
}

template<typename T,
         bool IsPointer,
         bool IsBinary   = false,
         typename Config = rocprim::default_config>
struct device_transform_benchmark : public benchmark_utils::autotune_interface
{

    std::string name() const override
    {

        using namespace std::string_literals;
        return bench_naming::format_name(
            "{lvl:device,algo:transform" + std::string(IsPointer ? "_pointer" : "")
            + ",op:" + std::string(IsBinary ? "binary" : "unary") + ",value_type:"
            + std::string(Traits<T>::name()) + ",cfg:" + transform_config_name<Config>() + "}");
    }

    void run(benchmark_utils::state&& state) override
    {
        const auto& stream = state.stream;
        const auto& bytes  = state.bytes;
        const auto& seed   = state.seed;

        using output_type = T;

        // Calculate the number of elements
        size_t size = bytes / sizeof(T);

        static constexpr bool debug_synchronous = false;

        // Generate data
        const auto           random_range = limit_random_range<T>(1, 100);
        const std::vector<T> input
            = get_random_data<T>(size, random_range.first, random_range.second, seed.get_0());

        common::device_ptr<T>           d_input(input);
        common::device_ptr<output_type> d_output(size);

        if constexpr(IsBinary)
        {
            const std::vector<T> input2
                = get_random_data<T>(size, random_range.first, random_range.second, seed.get_0());
            common::device_ptr<T> d_input2(input2);

            // If it is not a unary operator, it can not make use of the pointer optimization.
            const auto launch = [&]
            {
                auto transform_op = [](T v1, T v2) { return v1 + v2; };
                return rocprim::transform<Config>(rocprim::tuple(d_input.get(), d_input2.get()),
                                                  d_output.get(),
                                                  size,
                                                  transform_op,
                                                  stream,
                                                  debug_synchronous);
            };

            state.run([&] { HIP_CHECK(launch()); });
            state.set_throughput(size, sizeof(T) + sizeof(T));
        }
        else
        {
            const auto launch = [&]
            {
                auto transform_op = [](T v) { return v + T(5); };
                return rocprim::detail::transform_impl<IsPointer, Config>(d_input.get(),
                                                                          d_output.get(),
                                                                          size,
                                                                          transform_op,
                                                                          stream,
                                                                          debug_synchronous);
            };

            state.run([&] { HIP_CHECK(launch()); });
            state.set_throughput(size, sizeof(T));
        }
    }
};

template<typename T, bool IsPointer, unsigned int BlockSize, rocprim::cache_load_modifier LoadType>
struct device_transform_benchmark_generator
{

    template<unsigned int ItemsPerThread>
    struct create_ipt
    {
        using generated_config = rocprim::
            transform_config<BlockSize, 1 << ItemsPerThread, ROCPRIM_GRID_SIZE_LIMIT, LoadType>;

        void operator()(std::vector<std::unique_ptr<benchmark_utils::autotune_interface>>& storage)
        {
            storage.emplace_back(
                std::make_unique<
                    device_transform_benchmark<T, IsPointer, false, generated_config>>());
        }
    };

    static void create(std::vector<std::unique_ptr<benchmark_utils::autotune_interface>>& storage)
    {
        static constexpr unsigned int min_items_per_thread = 0;
        static constexpr unsigned int max_items_per_thread = rocprim::Log2<16>::VALUE;
        static_for_each<make_index_range<unsigned int, min_items_per_thread, max_items_per_thread>,
                        create_ipt>(storage);
    }
};

#endif // ROCPRIM_BENCHMARK_DEVICE_TRANSFORM_PARALLEL_HPP_
