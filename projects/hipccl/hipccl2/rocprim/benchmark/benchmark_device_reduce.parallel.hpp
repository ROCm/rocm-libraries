// MIT License
//
// Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include <cstddef>
#include <string>
#include <vector>

// Google Benchmark
#include <benchmark/benchmark.h>

// HIP API
#include <hip/hip_runtime.h>

// rocPRIM HIP API
#include <rocprim/rocprim.hpp>

#include "benchmark_utils.hpp"

#define HIP_CHECK(condition)         \
  {                                  \
    hipError_t error = condition;    \
    if(error != hipSuccess){         \
        std::cout << "HIP error: " << error << " line: " << __LINE__ << std::endl; \
        exit(error); \
    } \
  }

template<typename T              = int,
         typename BinaryFunction = rocprim::plus<T>,
         typename Config         = rocprim::detail::default_reduce_config<ROCPRIM_TARGET_ARCH, T>>
struct device_reduce_benchmark : public config_autotune_interface
{
    static std::string get_name_pattern()
    {
        return R"---((?P<algo>\S*)\<)---"
               R"---((?P<datatype>\S*),\s*reduce_config\<)---"
               R"---(\s*(?P<block_size>[0-9]+),\s*(?P<items_per_thread>[0-9]+)\>\>)---";
    }

    std::string name() const override
    {
        return std::string("device_reduce<" + std::string(Traits<T>::name()) + ", reduce_config<"
                           + pad_string(std::to_string(Config::block_size), 3) + ", "
                           + pad_string(std::to_string(Config::items_per_thread), 2) + ">>");
    }

    static constexpr unsigned int batch_size = 10;
    static constexpr unsigned int warmup_size = 5;

    void run(benchmark::State& state,
             size_t size,
             const hipStream_t stream) const override
    {
        BinaryFunction reduce_op{};
        std::vector<T> input = get_random_data<T>(size, T(0), T(1000));

        T * d_input;
        T * d_output;
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_input), size * sizeof(T)));
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_output), sizeof(T)));
        HIP_CHECK(
            hipMemcpy(
                d_input, input.data(),
                size * sizeof(T),
                hipMemcpyHostToDevice
                )
        );
        HIP_CHECK(hipDeviceSynchronize());

        // Allocate temporary storage memory
        size_t temp_storage_size_bytes;
        void * d_temp_storage = nullptr;
        // Get size of d_temp_storage
        HIP_CHECK(
            rocprim::reduce<Config>(
                d_temp_storage, temp_storage_size_bytes,
                d_input, d_output, T(), size,
                reduce_op, stream
                )
        );
        HIP_CHECK(hipMalloc(&d_temp_storage,temp_storage_size_bytes));
        HIP_CHECK(hipDeviceSynchronize());

        // Warm-up
        for(size_t i = 0; i < warmup_size; i++)
        {
            HIP_CHECK(
                rocprim::reduce<Config>(
                    d_temp_storage, temp_storage_size_bytes,
                    d_input, d_output, T(), size,
                    reduce_op, stream
                    )
            );
        }
        HIP_CHECK(hipDeviceSynchronize());

        for(auto _ : state)
        {
            auto start = std::chrono::high_resolution_clock::now();

            for(size_t i = 0; i < batch_size; i++)
            {
                HIP_CHECK(
                    rocprim::reduce<Config>(
                        d_temp_storage, temp_storage_size_bytes,
                        d_input, d_output, T(), size,
                        reduce_op, stream
                        )
                );
            }
            HIP_CHECK(hipStreamSynchronize(stream));

            auto end = std::chrono::high_resolution_clock::now();
            auto elapsed_seconds =
                std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
            state.SetIterationTime(elapsed_seconds.count());
        }
        state.SetBytesProcessed(state.iterations() * batch_size * size * sizeof(T));
        state.SetItemsProcessed(state.iterations() * batch_size * size);

        HIP_CHECK(hipFree(d_input));
        HIP_CHECK(hipFree(d_output));
        HIP_CHECK(hipFree(d_temp_storage));
    }
};

#endif
