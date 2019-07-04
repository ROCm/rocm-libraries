// MIT License
//
// Copyright (c) 2017 Advanced Micro Devices, Inc. All rights reserved.
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
#include "benchmark_utils.hpp"

// HIP API
#include <hip/hip_runtime.h>

// rocPRIM
#include <rocprim/rocprim.hpp>

#define HIP_CHECK(condition)         \
  {                                  \
    hipError_t error = condition;    \
    if(error != hipSuccess){         \
        std::cout << "HIP error: " << error << " line: " << __LINE__ << std::endl; \
        exit(error); \
    } \
  }

#ifndef DEFAULT_N
const size_t DEFAULT_N = 1024 * 1024 * 128;
#endif

enum class benchmark_kinds
{
    sort_keys,
    sort_pairs
};

namespace rp = rocprim;

template<
    class T,
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    unsigned int Trials
>
__global__
void sort_keys_kernel(const T * input, T * output)
{
    const unsigned int lid = hipThreadIdx_x;
    const unsigned int block_offset = hipBlockIdx_x * ItemsPerThread * BlockSize;

    T keys[ItemsPerThread];
    rp::block_load_direct_striped<BlockSize>(lid, input + block_offset, keys);

    #pragma nounroll
    for(unsigned int trial = 0; trial < Trials; trial++)
    {
        rp::block_radix_sort<T, BlockSize, ItemsPerThread> sort;
        sort.sort(keys);
    }

    rp::block_store_direct_striped<BlockSize>(lid, output + block_offset, keys);
}

template<
    class T,
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    unsigned int Trials
>
__global__
void sort_pairs_kernel(const T * input, T * output)
{
    const unsigned int lid = hipThreadIdx_x;
    const unsigned int block_offset = hipBlockIdx_x * ItemsPerThread * BlockSize;

    T keys[ItemsPerThread];
    T values[ItemsPerThread];
    rp::block_load_direct_striped<BlockSize>(lid, input + block_offset, keys);
    for(unsigned int i = 0; i < ItemsPerThread; i++)
    {
        values[i] = keys[i] + T(1);
    }

    #pragma nounroll
    for(unsigned int trial = 0; trial < Trials; trial++)
    {
        rp::block_radix_sort<T, BlockSize, ItemsPerThread, T> sort;
        sort.sort(keys, values);
    }

    for(unsigned int i = 0; i < ItemsPerThread; i++)
    {
        keys[i] += values[i];
    }
    rp::block_store_direct_striped<BlockSize>(lid, output + block_offset, keys);
}

template<
    class T,
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    unsigned int Trials = 10
>
void run_benchmark(benchmark::State& state, benchmark_kinds benchmark_kind, hipStream_t stream, size_t N)
{
    constexpr auto items_per_block = BlockSize * ItemsPerThread;
    const auto size = items_per_block * ((N + items_per_block - 1)/items_per_block);

    std::vector<T> input;
    if(std::is_floating_point<T>::value)
    {
        input = get_random_data<T>(size, (T)-1000, (T)+1000);
    }
    else
    {
        input = get_random_data<T>(
            size,
            std::numeric_limits<T>::min(),
            std::numeric_limits<T>::max()
        );
    }
    T * d_input;
    T * d_output;
    HIP_CHECK(hipMalloc(&d_input, size * sizeof(T)));
    HIP_CHECK(hipMalloc(&d_output, size * sizeof(T)));
    HIP_CHECK(
        hipMemcpy(
            d_input, input.data(),
            size * sizeof(T),
            hipMemcpyHostToDevice
        )
    );
    HIP_CHECK(hipDeviceSynchronize());

    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();

        if(benchmark_kind == benchmark_kinds::sort_keys)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(sort_keys_kernel<T, BlockSize, ItemsPerThread, Trials>),
                dim3(size/items_per_block), dim3(BlockSize), 0, stream,
                d_input, d_output
            );
        }
        else if(benchmark_kind == benchmark_kinds::sort_pairs)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(sort_pairs_kernel<T, BlockSize, ItemsPerThread, Trials>),
                dim3(size/items_per_block), dim3(BlockSize), 0, stream,
                d_input, d_output
            );
        }
        HIP_CHECK(hipPeekAtLastError());
        HIP_CHECK(hipDeviceSynchronize());

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }
    state.SetBytesProcessed(state.iterations() * Trials * size * sizeof(T));
    state.SetItemsProcessed(state.iterations() * Trials * size);

    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));
}

#define CREATE_BENCHMARK(T, BS, IPT) \
benchmark::RegisterBenchmark( \
    (std::string("block_radix_sort<" #T ", " #BS ", " #IPT ">.") + name).c_str(), \
    run_benchmark<T, BS, IPT>, \
    benchmark_kind, stream, size \
)

#define BENCHMARK_TYPE(type, block) \
    CREATE_BENCHMARK(type, block, 1), \
    CREATE_BENCHMARK(type, block, 2), \
    CREATE_BENCHMARK(type, block, 3), \
    CREATE_BENCHMARK(type, block, 4), \
    CREATE_BENCHMARK(type, block, 8)

void add_benchmarks(benchmark_kinds benchmark_kind,
                    const std::string& name,
                    std::vector<benchmark::internal::Benchmark*>& benchmarks,
                    hipStream_t stream,
                    size_t size)
{
    std::vector<benchmark::internal::Benchmark*> bs =
    {
        BENCHMARK_TYPE(int, 64),
        BENCHMARK_TYPE(int, 128),
        BENCHMARK_TYPE(int, 192),
        BENCHMARK_TYPE(int, 256),
        BENCHMARK_TYPE(int, 320),
        BENCHMARK_TYPE(int, 512),

        BENCHMARK_TYPE(int8_t, 64),
        BENCHMARK_TYPE(int8_t, 128),
        BENCHMARK_TYPE(int8_t, 192),
        BENCHMARK_TYPE(int8_t, 256),
        BENCHMARK_TYPE(int8_t, 320),
        BENCHMARK_TYPE(int8_t, 512),

        BENCHMARK_TYPE(uint8_t, 64),
        BENCHMARK_TYPE(uint8_t, 128),
        BENCHMARK_TYPE(uint8_t, 192),
        BENCHMARK_TYPE(uint8_t, 256),
        BENCHMARK_TYPE(uint8_t, 320),
        BENCHMARK_TYPE(uint8_t, 512),

        BENCHMARK_TYPE(rocprim::half, 64),
        BENCHMARK_TYPE(rocprim::half, 128),
        BENCHMARK_TYPE(rocprim::half, 192),
        BENCHMARK_TYPE(rocprim::half, 256),
        BENCHMARK_TYPE(rocprim::half, 320),
        BENCHMARK_TYPE(rocprim::half, 512),

        BENCHMARK_TYPE(long long, 64),
        BENCHMARK_TYPE(long long, 128),
        BENCHMARK_TYPE(long long, 192),
        BENCHMARK_TYPE(long long, 256),
        BENCHMARK_TYPE(long long, 320),
        BENCHMARK_TYPE(long long, 512),
    };

    benchmarks.insert(benchmarks.end(), bs.begin(), bs.end());
}

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

    // Add benchmarks
    std::vector<benchmark::internal::Benchmark*> benchmarks;
    add_benchmarks(benchmark_kinds::sort_keys, "sort(keys)", benchmarks, stream, size);
    add_benchmarks(benchmark_kinds::sort_pairs, "sort(keys, values)", benchmarks, stream, size);

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
