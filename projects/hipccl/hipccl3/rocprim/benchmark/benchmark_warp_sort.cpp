// MIT License
//
// Copyright (c) 2017-2019 Advanced Micro Devices, Inc. All rights reserved.
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

#include <iostream>
#include <chrono>
#include <vector>
#include <limits>
#include <string>
#include <cstdio>
#include <cstdlib>

// Google Benchmark
#include "benchmark/benchmark.h"
// CmdParser
#include "cmdparser.hpp"
// HIP API
#include <hip/hip_runtime.h>
// rocPRIM
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

#ifndef DEFAULT_N
const size_t DEFAULT_N = 1024 * 1024 * 32;
#endif

namespace rp = rocprim;

template<class K, unsigned int WarpSize, unsigned int Trials>
__global__
__launch_bounds__(ROCPRIM_DEFAULT_MAX_BLOCK_SIZE, ROCPRIM_DEFAULT_MIN_WARPS_PER_EU)
void warp_sort_kernel(K* input_key)
{
    const unsigned int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    auto key = input_key[i];
    rp::warp_sort<K, WarpSize> wsort;
    #pragma nounroll
    for(unsigned int trial = 0; trial < Trials; trial++)
    {
        wsort.sort(key);
    }
    input_key[i] = key;
}

template<class K, class V, unsigned int WarpSize, unsigned int Trials>
__global__
__launch_bounds__(ROCPRIM_DEFAULT_MAX_BLOCK_SIZE, ROCPRIM_DEFAULT_MIN_WARPS_PER_EU)
void warp_sort_by_key_kernel(K* input_key, V* input_value)
{
    const unsigned int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    auto key = input_key[i];
    auto value = input_value[i];
    rp::warp_sort<K, WarpSize, V> wsort;
     #pragma nounroll
    for(unsigned int trial = 0; trial < Trials; trial++)
    {
        wsort.sort(key, value);
    }
    input_key[i] = key;
    input_value[i] = value;
}

template<
    class Key,
    unsigned int BlockSize,
    unsigned int WarpSize,
    class Value = Key,
    bool SortByKey = false,
    unsigned int Trials = 100
>
void run_benchmark(benchmark::State& state, hipStream_t stream, size_t size)
{
    // Make sure size is a multiple of BlockSize
    size = BlockSize * ((size + BlockSize - 1)/BlockSize);
    // Allocate and fill memory
    std::vector<Key> input_key = get_random_data(size, Key(0), Key(10000));
    std::vector<Value> input_value(size_t(1));
    if(SortByKey) input_value = get_random_data(size, Value(0), Value(10000));
    Key * d_input_key = nullptr;
    Value * d_input_value = nullptr;
    HIP_CHECK(hipMalloc(&d_input_key, size * sizeof(Key)));
    if(SortByKey) HIP_CHECK(hipMalloc(&d_input_value, size * sizeof(Value)));
    HIP_CHECK(
        hipMemcpy(
            d_input_key, input_key.data(),
            size * sizeof(Key),
            hipMemcpyHostToDevice
        )
    );
    if(SortByKey) HIP_CHECK(
        hipMemcpy(
            d_input_value, input_value.data(),
            size * sizeof(Value),
            hipMemcpyHostToDevice
        )
    );
    HIP_CHECK(hipDeviceSynchronize());

    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();
        if(SortByKey)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(warp_sort_by_key_kernel<Key, Value, WarpSize, Trials>),
                dim3(size/BlockSize), dim3(BlockSize), 0, stream,
                d_input_key, d_input_value
            );
        }
        else
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(warp_sort_kernel<Key, WarpSize, Trials>),
                dim3(size/BlockSize), dim3(BlockSize), 0, stream,
                d_input_key
            );
        }
        HIP_CHECK(hipPeekAtLastError());
        HIP_CHECK(hipDeviceSynchronize());

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }
    // SortByKey also transfers values
    auto sorted_type_size = sizeof(Key);
    if(SortByKey) sorted_type_size += sizeof(Value);
    state.SetBytesProcessed(state.iterations() * size * sorted_type_size * Trials);
    state.SetItemsProcessed(state.iterations() * size * Trials);

    HIP_CHECK(hipFree(d_input_key));
    HIP_CHECK(hipFree(d_input_value));
}

#define CREATE_SORT_BENCHMARK(K, BS, WS) \
    benchmark::RegisterBenchmark( \
        "warp_sort<"#K", "#BS", "#WS">.sort(only keys)", \
        run_benchmark<K, BS, WS>, \
        stream, size \
    )

#define CREATE_SORTBYKEY_BENCHMARK(K, V, BS, WS) \
    benchmark::RegisterBenchmark( \
        "warp_sort<"#K", "#BS", "#WS", "#V">.sort", \
        run_benchmark<K, BS, WS, V, true>, \
        stream, size \
    )

#define BENCHMARK_TYPE(type) \
    CREATE_SORT_BENCHMARK(type, 64, 64), \
    CREATE_SORT_BENCHMARK(type, 128, 64), \
    CREATE_SORT_BENCHMARK(type, 256, 64), \
    CREATE_SORT_BENCHMARK(type, 64, 32), \
    CREATE_SORT_BENCHMARK(type, 64, 16)

#define BENCHMARK_KEY_TYPE(type, value) \
    CREATE_SORTBYKEY_BENCHMARK(type, value, 64, 64), \
    CREATE_SORTBYKEY_BENCHMARK(type, value, 256, 64)

int main(int argc, char *argv[])
{
    cli::Parser parser(argc, argv);
    parser.set_optional<size_t>("size", "size", DEFAULT_N, "number of values");
    parser.set_optional<int>("trials", "trials", -1, "number of iterations");
    parser.run_and_exit_if_error();

    // Parse argv
    benchmark::Initialize(&argc, argv);
    const size_t size = parser.get<size_t>("size");
    const int trials = parser.get<int>("trials");

    // HIP
    hipStream_t stream = 0; // default
    hipDeviceProp_t devProp;
    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    HIP_CHECK(hipGetDeviceProperties(&devProp, device_id));
    std::cout << "[HIP] Device name: " << devProp.name << std::endl;

    using custom_double2 = custom_type<double, double>;
    using custom_int_double = custom_type<int, double>;
    std::vector<benchmark::internal::Benchmark*> benchmarks =
    {
        BENCHMARK_TYPE(int),
        BENCHMARK_TYPE(float),
        BENCHMARK_TYPE(double),
        BENCHMARK_TYPE(int8_t),
        BENCHMARK_TYPE(uint8_t),
        BENCHMARK_TYPE(rocprim::half),

        BENCHMARK_KEY_TYPE(float, float),
        BENCHMARK_KEY_TYPE(unsigned int, int),
        BENCHMARK_KEY_TYPE(int, custom_double2),
        BENCHMARK_KEY_TYPE(int, custom_int_double),
        BENCHMARK_KEY_TYPE(int8_t, int8_t),
        BENCHMARK_KEY_TYPE(uint8_t, uint8_t),
        BENCHMARK_KEY_TYPE(rocprim::half, rocprim::half)
    };

    // Use manual timing
    for(auto& b : benchmarks)
    {
        b->UseManualTime();
        b->Unit(benchmark::kMillisecond);
    }

    // Force number of iterations
    if(trials > 0)
    {
        for(auto& b : benchmarks)
        {
            b->Iterations(trials);
        }
    }

    // Run benchmarks
    benchmark::RunSpecifiedBenchmarks();
    return 0;
}
