// MIT License
//
// Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_BENCHMARK_DEVICE_REDUCE_PARALLEL_HPP_
#define ROCPRIM_BENCHMARK_DEVICE_REDUCE_PARALLEL_HPP_

#include "benchmark_utils.hpp"

// Google Benchmark
#include <benchmark/benchmark.h>

// HIP API
#include <hip/hip_runtime.h>

// rocPRIM HIP API
#include <rocprim/block/block_reduce.hpp>
#include <rocprim/device/config_types.hpp>
#include <rocprim/device/detail/device_config_helper.hpp>
#include <rocprim/device/device_reduce.hpp>
#include <rocprim/functional.hpp>

#include <cstddef>
#include <string>
#include <vector>

constexpr const char* get_reduce_method_name(rocprim::block_reduce_algorithm alg)
{
    switch(alg)
    {
        case rocprim::block_reduce_algorithm::raking_reduce: return "raking_reduce";
        case rocprim::block_reduce_algorithm::raking_reduce_commutative_only:
            return "raking_reduce_commutative_only";
        case rocprim::block_reduce_algorithm::using_warp_reduce:
            return "using_warp_reduce";
            // Not using `default: ...` because it kills effectiveness of -Wswitch
    }
    return "unknown_algorithm";
}

template<typename Config>
std::string config_name()
{
    const rocprim::detail::reduce_config_params config = Config();
    return "{bs:" + std::to_string(config.reduce_config.block_size)
           + ",ipt:" + std::to_string(config.reduce_config.items_per_thread)
           + ",method:" + std::string(get_reduce_method_name(config.block_reduce_method)) + "}";
}

template<>
inline std::string config_name<rocprim::default_config>()
{
    return "default_config";
}

template<typename T              = int,
         typename BinaryFunction = rocprim::plus<T>,
         typename Config         = rocprim::default_config>
struct device_reduce_benchmark : public benchmark_utils::autotune_interface
{
    std::string name() const override
    {
        return bench_naming::format_name("{lvl:device,algo:reduce,key_type:"
                                         + std::string(Traits<T>::name())
                                         + ",cfg:" + config_name<Config>() + "}");
    }

    void run(benchmark::State& gbench_state, benchmark_utils::state& state) const override
    {
        const auto& stream = state.stream;
        const auto& bytes  = state.bytes;
        const auto& seed   = state.seed;

        // Calculate the number of elements
        size_t size = bytes / sizeof(T);

        BinaryFunction reduce_op{};
        const auto     random_range = limit_random_range<T>(0, 1000);
        std::vector<T> input
            = get_random_data<T>(size, random_range.first, random_range.second, seed.get_0());

        T* d_input;
        T* d_output;
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_input), size * sizeof(T)));
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_output), sizeof(T)));
        HIP_CHECK(hipMemcpy(d_input, input.data(), size * sizeof(T), hipMemcpyHostToDevice));
        HIP_CHECK(hipDeviceSynchronize());

        // Allocate temporary storage memory
        size_t temp_storage_size_bytes;
        void*  d_temp_storage = nullptr;
        // Get size of d_temp_storage
        HIP_CHECK(rocprim::reduce<Config>(d_temp_storage,
                                          temp_storage_size_bytes,
                                          d_input,
                                          d_output,
                                          T(),
                                          size,
                                          reduce_op,
                                          stream));
        HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_size_bytes));
        HIP_CHECK(hipDeviceSynchronize());

        state.run(gbench_state,
                  [&]
                  {
                      HIP_CHECK(rocprim::reduce<Config>(d_temp_storage,
                                                        temp_storage_size_bytes,
                                                        d_input,
                                                        d_output,
                                                        T(),
                                                        size,
                                                        reduce_op,
                                                        stream));
                  });
        state.set_items_processed_per_iteration<T>(gbench_state, size);

        HIP_CHECK(hipFree(d_input));
        HIP_CHECK(hipFree(d_output));
        HIP_CHECK(hipFree(d_temp_storage));
    }
};

#endif
