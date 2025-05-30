// MIT License
//
// Copyright (c) 2019-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_BENCHMARK_DEVICE_ADJACENT_DIFFERENCE_PARALLEL_HPP_
#define ROCPRIM_BENCHMARK_DEVICE_ADJACENT_DIFFERENCE_PARALLEL_HPP_

#include "benchmark_utils.hpp"

#include "../common/device_adjacent_difference.hpp"
#include "../common/utils_device_ptr.hpp"

// Google Benchmark
#include <benchmark/benchmark.h>

// HIP API
#include <hip/hip_runtime_api.h>

// rocPRIM
#include <rocprim/config.hpp>
#include <rocprim/detail/various.hpp>
#include <rocprim/device/config_types.hpp>
#include <rocprim/device/detail/device_config_helper.hpp>
#include <rocprim/functional.hpp>

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

template<typename Config>
std::string config_name()
{
    auto config = Config();
    return "{bs:" + std::to_string(config.block_size)
           + ",ipt:" + std::to_string(config.items_per_thread) + "}";
}

template<>
inline std::string config_name<rocprim::default_config>()
{
    return "default_config";
}

template<typename T                   = int,
         bool                Left     = false,
         common::api_variant Aliasing = common::api_variant::no_alias,
         typename Config              = rocprim::default_config>
struct device_adjacent_difference_benchmark : public benchmark_utils::autotune_interface
{
    std::string name() const override
    {

        using namespace std::string_literals;
        return bench_naming::format_name(
            "{lvl:device,algo:adjacent_difference" + (Left ? ""s : "_right"s)
            + (Aliasing == common::api_variant::no_alias ? ""s : "_inplace"s) + ",value_type:"
            + std::string(Traits<T>::name()) + ",cfg:" + config_name<Config>() + "}");
    }

    void run(benchmark_utils::state&& state) override
    {
        const auto& stream = state.stream;
        const auto& bytes  = state.bytes;
        const auto& seed   = state.seed;

        using output_type = T;

        static constexpr bool debug_synchronous = false;

        // Generate data
        const size_t         size         = bytes / sizeof(T);
        const auto           random_range = limit_random_range<T>(1, 100);
        const std::vector<T> input
            = get_random_data<T>(size, random_range.first, random_range.second, seed.get_0());

        common::device_ptr<T>           d_input(input);
        common::device_ptr<output_type> d_output;

        if constexpr(Aliasing == common::api_variant::no_alias)
        {
            d_output.resize(size);
        }

        static constexpr auto left_tag  = rocprim::detail::bool_constant<Left>{};
        static constexpr auto alias_tag = std::integral_constant<common::api_variant, Aliasing>{};

        // Allocate temporary storage
        std::size_t              temp_storage_size;
        common::device_ptr<void> d_temp_storage;

        const auto launch = [&]
        {
            return common::dispatch_adjacent_difference(left_tag,
                                                        alias_tag,
                                                        d_temp_storage.get(),
                                                        temp_storage_size,
                                                        d_input.get(),
                                                        d_output.get(),
                                                        size,
                                                        rocprim::plus<>{},
                                                        stream,
                                                        debug_synchronous);
        };
        HIP_CHECK(launch());
        d_temp_storage.resize(temp_storage_size);

        state.run([&] { HIP_CHECK(launch()); });

        state.set_throughput(size, sizeof(T));
    }
};

template<typename T, unsigned int BlockSize, bool Left, common::api_variant Aliasing>
struct device_adjacent_difference_benchmark_generator
{
    // Device Adjacent difference uses block_load/store_transpose to coalesce memory transaction to global memory
    // However it accesses shared memory with a stride of items per thread, which leads to reduced performance if power
    // of two is used for small types. Experiments shown that primes are the best choice for performance.
    static constexpr std::array<int, 12> primes{1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31};

    static constexpr unsigned int max_items_per_thread_arg
        = TUNING_SHARED_MEMORY_MAX / (BlockSize * sizeof(T) * 2 + sizeof(T));

    template<unsigned int IptValueIndex>
    struct create_ipt
    {
        template<int ipt_num = primes[IptValueIndex]>
        auto operator()(std::vector<std::unique_ptr<benchmark_utils::autotune_interface>>& storage)
            -> std::enable_if_t<(ipt_num < max_items_per_thread_arg)>
        {
            using generated_config = rocprim::adjacent_difference_config<BlockSize, ipt_num>;

            storage.emplace_back(
                std::make_unique<
                    device_adjacent_difference_benchmark<T, Left, Aliasing, generated_config>>());
        }

        template<int ipt_num = primes[IptValueIndex]>
        auto operator()(std::vector<std::unique_ptr<benchmark_utils::autotune_interface>>&)
            -> std::enable_if_t<!(ipt_num < max_items_per_thread_arg)>
        {}
    };

    static void create(std::vector<std::unique_ptr<benchmark_utils::autotune_interface>>& storage)
    {
        static_for_each<make_index_range<unsigned int, 0, primes.size() - 1>, create_ipt>(storage);
    }
};

#endif // ROCPRIM_BENCHMARK_DEVICE_ADJACENT_DIFFERENCE_PARALLEL_HPP_
