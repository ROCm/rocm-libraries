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
struct device_adjacent_difference_benchmark : public config_autotune_interface
{

    std::string name() const override
    {

        using namespace std::string_literals;
        return bench_naming::format_name(
            "{lvl:device,algo:adjacent_difference" + (Left ? ""s : "_right"s)
            + (Aliasing == common::api_variant::no_alias ? ""s : "_inplace"s) + ",value_type:"
            + std::string(Traits<T>::name()) + ",cfg:" + config_name<Config>() + "}");
    }

    static constexpr unsigned int batch_size  = 10;
    static constexpr unsigned int warmup_size = 5;

    void run(benchmark::State&   state,
             const std::size_t   bytes,
             const managed_seed& seed,
             hipStream_t         stream) const override
    {
        using output_type = T;

        static constexpr bool debug_synchronous = false;

        // Generate data
        const size_t         size         = bytes / sizeof(T);
        const auto           random_range = limit_random_range<T>(1, 100);
        const std::vector<T> input
            = get_random_data<T>(size, random_range.first, random_range.second, seed.get_0());

        T*           d_input;
        output_type* d_output = nullptr;
        HIP_CHECK(hipMalloc(&d_input, input.size() * sizeof(input[0])));
        HIP_CHECK(hipMemcpy(d_input,
                            input.data(),
                            input.size() * sizeof(input[0]),
                            hipMemcpyHostToDevice));

        if ROCPRIM_IF_CONSTEXPR(Aliasing == common::api_variant::no_alias)
        {
            HIP_CHECK(hipMalloc(&d_output, size * sizeof(output_type)));
        }

        static constexpr auto left_tag  = rocprim::detail::bool_constant<Left>{};
        static constexpr auto alias_tag = std::integral_constant<common::api_variant, Aliasing>{};

        // Allocate temporary storage
        std::size_t temp_storage_size;
        void*       d_temp_storage = nullptr;

        const auto launch = [&]
        {
            return common::dispatch_adjacent_difference(left_tag,
                                                        alias_tag,
                                                        d_temp_storage,
                                                        temp_storage_size,
                                                        d_input,
                                                        d_output,
                                                        size,
                                                        rocprim::plus<>{},
                                                        stream,
                                                        debug_synchronous);
        };
        HIP_CHECK(launch());
        HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_size));

        // Warm-up
        for(size_t i = 0; i < warmup_size; ++i)
        {
            HIP_CHECK(launch());
        }
        HIP_CHECK(hipDeviceSynchronize());

        // HIP events creation
        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));

        // Run
        for(auto _ : state)
        {
            // Record start event
            HIP_CHECK(hipEventRecord(start, stream));

            for(size_t i = 0; i < batch_size; ++i)
            {
                HIP_CHECK(launch());
            }

            // Record stop event and wait until it completes
            HIP_CHECK(hipEventRecord(stop, stream));
            HIP_CHECK(hipEventSynchronize(stop));

            float elapsed_mseconds;
            HIP_CHECK(hipEventElapsedTime(&elapsed_mseconds, start, stop));
            state.SetIterationTime(elapsed_mseconds / 1000);
        }

        // Destroy HIP events
        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));

        state.SetBytesProcessed(state.iterations() * batch_size * size * sizeof(T));
        state.SetItemsProcessed(state.iterations() * batch_size * size);

        HIP_CHECK(hipFree(d_input));
        if ROCPRIM_IF_CONSTEXPR(Aliasing == common::api_variant::no_alias)
        {
            HIP_CHECK(hipFree(d_output));
        }
        HIP_CHECK(hipFree(d_temp_storage));
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
        auto operator()(std::vector<std::unique_ptr<config_autotune_interface>>& storage)
            -> std::enable_if_t<(ipt_num < max_items_per_thread_arg)>
        {
            using generated_config = rocprim::adjacent_difference_config<BlockSize, ipt_num>;

            storage.emplace_back(
                std::make_unique<
                    device_adjacent_difference_benchmark<T, Left, Aliasing, generated_config>>());
        }

        template<int ipt_num = primes[IptValueIndex]>
        auto operator()(std::vector<std::unique_ptr<config_autotune_interface>>&)
            -> std::enable_if_t<!(ipt_num < max_items_per_thread_arg)>
        {}
    };

    static void create(std::vector<std::unique_ptr<config_autotune_interface>>& storage)
    {
        static_for_each<make_index_range<unsigned int, 0, primes.size() - 1>, create_ipt>(storage);
    }
};

#endif // ROCPRIM_BENCHMARK_DEVICE_ADJACENT_DIFFERENCE_PARALLEL_HPP_
