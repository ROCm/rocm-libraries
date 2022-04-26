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
#include <type_traits>
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
    sort_pairs,
    stable_sort
};

namespace rp = rocprim;

template<
    class T,
    unsigned int BlockSize,
    unsigned int ItemsPerThread
>
__global__
__launch_bounds__(BlockSize)
void sort_keys_kernel(const T * input, T * output)
{

    const unsigned int lid = threadIdx.x;
    const unsigned int block_offset = blockIdx.x * ItemsPerThread * BlockSize;

    T keys[ItemsPerThread];
    rp::block_load_direct_striped<BlockSize>(lid, input + block_offset, keys);

    rp::block_sort<T, BlockSize, ItemsPerThread> bsort;
    bsort.sort(keys);

    rp::block_store_direct_blocked(lid, output + block_offset, keys);
}

template<
    class T,
    unsigned int BlockSize,
    unsigned int ItemsPerThread
>
__global__
__launch_bounds__(BlockSize)
void sort_pairs_kernel(const T * input, T * output)
{
    const unsigned int lid = threadIdx.x;
    const unsigned int block_offset = blockIdx.x * ItemsPerThread * BlockSize;

    T keys[ItemsPerThread];
    T values[ItemsPerThread];
    rp::block_load_direct_striped<BlockSize>(lid, input + block_offset, keys);

    ROCPRIM_UNROLL
    for(unsigned int item = 0; item < ItemsPerThread; ++item) {
        values[item] = keys[item] + T(1);
    }

    rp::block_sort<T, BlockSize, ItemsPerThread, T> bsort;
    bsort.sort(keys, values);

    ROCPRIM_UNROLL
    for(unsigned int item = 0; item < ItemsPerThread; ++item) {
        keys[item] = keys[item] + values[item];
    }

    rp::block_store_direct_blocked(lid, output + block_offset, keys);
}

template<
    class T,
    unsigned int BlockSize,
    unsigned int ItemsPerThread
>
__global__
__launch_bounds__(BlockSize)
void stable_sort_kernel(const T* input, T* output)
{
    const unsigned int lid = threadIdx.x;
    const unsigned int block_offset = blockIdx.x * ItemsPerThread * BlockSize;

    T keys[ItemsPerThread];
    rp::block_load_direct_striped<BlockSize>(lid, input + block_offset, keys);

    using stable_key_type = rocprim::tuple<T, unsigned int>;
    stable_key_type stable_keys[ItemsPerThread];

    ROCPRIM_UNROLL
    for(unsigned int item = 0; item < ItemsPerThread; ++item) {
        stable_keys[item] = rp::make_tuple(keys[item], ItemsPerThread * lid + item);
    }

    // Special comparison that preserves relative order of equal keys
    auto stable_compare_function
        = [](const stable_key_type& a, const stable_key_type& b) mutable -> bool {
        const bool ab = rp::less<T> {}(rp::get<0>(a), rp::get<0>(b));
        const bool ba = rp::less<T> {}(rp::get<0>(b), rp::get<0>(a));
        return ab || (!ba && (rp::get<1>(a) < rp::get<1>(b)));
    };

    rp::block_sort<stable_key_type, BlockSize, ItemsPerThread> bsort;
    bsort.sort(stable_keys, stable_compare_function);

    ROCPRIM_UNROLL
    for(unsigned int item = 0; item < ItemsPerThread; ++item) {
        keys[item] = rp::get<0>(stable_keys[item]);
    }

    rp::block_store_direct_blocked(lid, output + block_offset, keys);
}

template<
    class T,
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    unsigned int Trials = 10
>
void run_benchmark(benchmark::State& state, benchmark_kinds benchmark_kind, hipStream_t stream, size_t N)
{
    static constexpr auto block_size = BlockSize;
    static constexpr auto items_per_block = BlockSize * ItemsPerThread;
    const auto size = items_per_block * ((N + items_per_block - 1) / items_per_block);

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
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_input), size * sizeof(T)));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_output), size * sizeof(T)));
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
            ROCPRIM_NO_UNROLL
            for(unsigned int trial = 0; trial < Trials; ++trial) {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(sort_keys_kernel<T, block_size, ItemsPerThread>),
                    dim3(size / items_per_block), dim3(block_size), 0, stream,
                    d_input, d_output
                );
            }
        }
        else if(benchmark_kind == benchmark_kinds::sort_pairs)
        {
            ROCPRIM_NO_UNROLL
            for(unsigned int trial = 0; trial < Trials; ++trial) {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(sort_pairs_kernel<T, block_size, ItemsPerThread>),
                    dim3(size / items_per_block), dim3(block_size), 0, stream,
                    d_input, d_output
                );
            }
        }
        else if(benchmark_kind == benchmark_kinds::stable_sort)
        {
            ROCPRIM_NO_UNROLL
            for(unsigned int trial = 0; trial < Trials; ++trial) {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(stable_sort_kernel<T, block_size, ItemsPerThread>),
                    dim3(size / items_per_block), dim3(block_size), 0, stream,
                    d_input, d_output
                );
            }
        }
        HIP_CHECK(hipGetLastError());
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

#define CREATE_BENCHMARK_ITEMS(T, BS, ITEMS) \
benchmark::RegisterBenchmark( \
    (std::string("block_sort<" #T ", " #BS ", " #ITEMS ">.") + name).c_str(), \
    run_benchmark<T, BS, ITEMS>, \
    benchmark_kind, stream, size \
)

#define CREATE_BENCHMARK(T, BS) CREATE_BENCHMARK_ITEMS(T, BS, 1)

void add_benchmarks(benchmark_kinds benchmark_kind,
                    const std::string& name,
                    std::vector<benchmark::internal::Benchmark*>& benchmarks,
                    hipStream_t stream,
                    size_t size)
{
    std::vector<benchmark::internal::Benchmark*> bs =
    {
        CREATE_BENCHMARK_ITEMS(int, 64, 1),
        CREATE_BENCHMARK_ITEMS(int, 64,  2),
        CREATE_BENCHMARK_ITEMS(int, 128, 1),
        CREATE_BENCHMARK_ITEMS(int, 64,  4),
        CREATE_BENCHMARK_ITEMS(int, 128, 2),
        CREATE_BENCHMARK_ITEMS(int, 256, 1),
        CREATE_BENCHMARK_ITEMS(int, 64,  8),
        CREATE_BENCHMARK_ITEMS(int, 128, 4),
        CREATE_BENCHMARK_ITEMS(int, 256, 2),
        CREATE_BENCHMARK_ITEMS(int, 512, 1),
        CREATE_BENCHMARK_ITEMS(int, 128,  8),
        CREATE_BENCHMARK_ITEMS(int, 256,  4),
        CREATE_BENCHMARK_ITEMS(int, 512,  2),
        CREATE_BENCHMARK_ITEMS(int, 1024, 1),
        CREATE_BENCHMARK_ITEMS(int, 256, 8),
        CREATE_BENCHMARK_ITEMS(int, 512, 4),
        CREATE_BENCHMARK_ITEMS(int, 1024, 2),
        CREATE_BENCHMARK_ITEMS(int, 512, 8),
        CREATE_BENCHMARK_ITEMS(int, 1024, 4),
        CREATE_BENCHMARK_ITEMS(int, 1024, 8),
        
        CREATE_BENCHMARK_ITEMS(int8_t, 64, 1),
        CREATE_BENCHMARK_ITEMS(int8_t, 64,  2),
        CREATE_BENCHMARK_ITEMS(int8_t, 128, 1),
        CREATE_BENCHMARK_ITEMS(int8_t, 64,  4),
        CREATE_BENCHMARK_ITEMS(int8_t, 128, 2),
        CREATE_BENCHMARK_ITEMS(int8_t, 256, 1),
        CREATE_BENCHMARK_ITEMS(int8_t, 64,  8),
        CREATE_BENCHMARK_ITEMS(int8_t, 128, 4),
        CREATE_BENCHMARK_ITEMS(int8_t, 256, 2),
        CREATE_BENCHMARK_ITEMS(int8_t, 512, 1),
        CREATE_BENCHMARK_ITEMS(int8_t, 128,  8),
        CREATE_BENCHMARK_ITEMS(int8_t, 256,  4),
        CREATE_BENCHMARK_ITEMS(int8_t, 512,  2),
        CREATE_BENCHMARK_ITEMS(int8_t, 1024, 1),
        CREATE_BENCHMARK_ITEMS(int8_t, 256, 8),
        CREATE_BENCHMARK_ITEMS(int8_t, 512, 4),
        CREATE_BENCHMARK_ITEMS(int8_t, 1024, 2),
        CREATE_BENCHMARK_ITEMS(int8_t, 512, 8),
        CREATE_BENCHMARK_ITEMS(int8_t, 1024, 4),
        CREATE_BENCHMARK_ITEMS(int8_t, 1024, 8),

        CREATE_BENCHMARK_ITEMS(uint8_t, 64, 1),
        CREATE_BENCHMARK_ITEMS(uint8_t, 64,  2),
        CREATE_BENCHMARK_ITEMS(uint8_t, 128, 1),
        CREATE_BENCHMARK_ITEMS(uint8_t, 64,  4),
        CREATE_BENCHMARK_ITEMS(uint8_t, 128, 2),
        CREATE_BENCHMARK_ITEMS(uint8_t, 256, 1),
        CREATE_BENCHMARK_ITEMS(uint8_t, 64,  8),
        CREATE_BENCHMARK_ITEMS(uint8_t, 128, 4),
        CREATE_BENCHMARK_ITEMS(uint8_t, 256, 2),
        CREATE_BENCHMARK_ITEMS(uint8_t, 512, 1),
        CREATE_BENCHMARK_ITEMS(uint8_t, 128,  8),
        CREATE_BENCHMARK_ITEMS(uint8_t, 256,  4),
        CREATE_BENCHMARK_ITEMS(uint8_t, 512,  2),
        CREATE_BENCHMARK_ITEMS(uint8_t, 1024, 1),
        CREATE_BENCHMARK_ITEMS(uint8_t, 256, 8),
        CREATE_BENCHMARK_ITEMS(uint8_t, 512, 4),
        CREATE_BENCHMARK_ITEMS(uint8_t, 1024, 2),
        CREATE_BENCHMARK_ITEMS(uint8_t, 512, 8),
        CREATE_BENCHMARK_ITEMS(uint8_t, 1024, 4),
        CREATE_BENCHMARK_ITEMS(uint8_t, 1024, 8),

        CREATE_BENCHMARK_ITEMS(rocprim::half, 64, 1),
        CREATE_BENCHMARK_ITEMS(rocprim::half, 64,  2),
        CREATE_BENCHMARK_ITEMS(rocprim::half, 128, 1),
        CREATE_BENCHMARK_ITEMS(rocprim::half, 64,  4),
        CREATE_BENCHMARK_ITEMS(rocprim::half, 128, 2),
        CREATE_BENCHMARK_ITEMS(rocprim::half, 256, 1),
        CREATE_BENCHMARK_ITEMS(rocprim::half, 64,  8),
        CREATE_BENCHMARK_ITEMS(rocprim::half, 128, 4),
        CREATE_BENCHMARK_ITEMS(rocprim::half, 256, 2),
        CREATE_BENCHMARK_ITEMS(rocprim::half, 512, 1),
        CREATE_BENCHMARK_ITEMS(rocprim::half, 128,  8),
        CREATE_BENCHMARK_ITEMS(rocprim::half, 256,  4),
        CREATE_BENCHMARK_ITEMS(rocprim::half, 512,  2),
        CREATE_BENCHMARK_ITEMS(rocprim::half, 1024, 1),
        CREATE_BENCHMARK_ITEMS(rocprim::half, 256, 8),
        CREATE_BENCHMARK_ITEMS(rocprim::half, 512, 4),
        CREATE_BENCHMARK_ITEMS(rocprim::half, 1024, 2),
        CREATE_BENCHMARK_ITEMS(rocprim::half, 512, 8),
        CREATE_BENCHMARK_ITEMS(rocprim::half, 1024, 4),
        CREATE_BENCHMARK_ITEMS(rocprim::half, 1024, 8),

        CREATE_BENCHMARK_ITEMS(long long, 64, 1),
        CREATE_BENCHMARK_ITEMS(long long, 64,  2),
        CREATE_BENCHMARK_ITEMS(long long, 128, 1),
        CREATE_BENCHMARK_ITEMS(long long, 64,  4),
        CREATE_BENCHMARK_ITEMS(long long, 128, 2),
        CREATE_BENCHMARK_ITEMS(long long, 256, 1),
        CREATE_BENCHMARK_ITEMS(long long, 64,  8),
        CREATE_BENCHMARK_ITEMS(long long, 128, 4),
        CREATE_BENCHMARK_ITEMS(long long, 256, 2),
        CREATE_BENCHMARK_ITEMS(long long, 512, 1),
        CREATE_BENCHMARK_ITEMS(long long, 128,  8),
        CREATE_BENCHMARK_ITEMS(long long, 256,  4),
        CREATE_BENCHMARK_ITEMS(long long, 512,  2),
        CREATE_BENCHMARK_ITEMS(long long, 1024, 1),
        CREATE_BENCHMARK_ITEMS(long long, 256, 8),
        CREATE_BENCHMARK_ITEMS(long long, 512, 4),
        CREATE_BENCHMARK_ITEMS(long long, 1024, 2),
        CREATE_BENCHMARK_ITEMS(long long, 512, 8),
        CREATE_BENCHMARK_ITEMS(long long, 1024, 4)
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

    // Benchmark info
    add_common_benchmark_info();
    benchmark::AddCustomContext("size", std::to_string(size));

    // Add benchmarks
    std::vector<benchmark::internal::Benchmark*> benchmarks;
    add_benchmarks(benchmark_kinds::sort_keys,   "sort(keys)", benchmarks, stream, size);
    add_benchmarks(benchmark_kinds::sort_pairs,  "sort(keys, values)", benchmarks, stream, size);
    add_benchmarks(benchmark_kinds::stable_sort, "stable_sort(keys)", benchmarks, stream, size);

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
