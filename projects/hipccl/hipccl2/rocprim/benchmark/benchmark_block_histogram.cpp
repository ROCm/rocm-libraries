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

namespace rp = rocprim;

template<
    class Runner,
    class T,
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    unsigned int BinSize,
    unsigned int Trials
>
__global__
__launch_bounds__(BlockSize, ROCPRIM_DEFAULT_MIN_WARPS_PER_EU)
void kernel(const T* input, T* output)
{
    Runner::template run<T, BlockSize, ItemsPerThread, BinSize, Trials>(input, output);
}

template<rocprim::block_histogram_algorithm algorithm>
struct histogram
{
    template<
        class T,
        unsigned int BlockSize,
        unsigned int ItemsPerThread,
        unsigned int BinSize,
        unsigned int Trials
    >
    __device__
    static void run(const T* input, T* output)
    {
        const unsigned int index = ((hipBlockIdx_x * BlockSize) + hipThreadIdx_x) * ItemsPerThread;
        unsigned int global_offset = hipBlockIdx_x * BinSize;

        T values[ItemsPerThread];
        for(unsigned int k = 0; k < ItemsPerThread; k++)
        {
            values[k] = input[index + k];
        }

        using bhistogram_t =  rp::block_histogram<T, BlockSize, ItemsPerThread, BinSize, algorithm>;
        __shared__ T histogram[BinSize];
        __shared__ typename bhistogram_t::storage_type storage;

        #pragma nounroll
        for(unsigned int trial = 0; trial < Trials; trial++)
        {
            bhistogram_t().histogram(values, histogram, storage);
        }

        #pragma unroll
        for (unsigned int offset = 0; offset < BinSize; offset += BlockSize)
        {
            if(offset + hipThreadIdx_x < BinSize)
            {
                output[global_offset + hipThreadIdx_x] = histogram[offset + hipThreadIdx_x];
                global_offset += BlockSize;
            }
        }
    }
};

template<
    class Benchmark,
    class T,
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    unsigned int BinSize = BlockSize,
    unsigned int Trials = 100
>
void run_benchmark(benchmark::State& state, hipStream_t stream, size_t N)
{
    // Make sure size is a multiple of BlockSize
    constexpr auto items_per_block = BlockSize * ItemsPerThread;
    const auto size = items_per_block * ((N + items_per_block - 1)/items_per_block);
    const auto bin_size = BinSize * ((N + items_per_block - 1)/items_per_block);
    // Allocate and fill memory
    std::vector<T> input(size, 0.0f);
    T * d_input;
    T * d_output;
    HIP_CHECK(hipMalloc(&d_input, size * sizeof(T)));
    HIP_CHECK(hipMalloc(&d_output, bin_size * sizeof(T)));
    HIP_CHECK(
        hipMemcpy(
            d_input, input.data(),
            size * sizeof(T),
            hipMemcpyHostToDevice
        )
    );
    HIP_CHECK(hipDeviceSynchronize());

    for (auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(kernel<Benchmark, T, BlockSize, ItemsPerThread, BinSize, Trials>),
            dim3(size/items_per_block), dim3(BlockSize), 0, stream,
            d_input, d_output
        );
        HIP_CHECK(hipPeekAtLastError());
        HIP_CHECK(hipDeviceSynchronize());

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

        state.SetIterationTime(elapsed_seconds.count());
    }
    state.SetBytesProcessed(state.iterations() * size * sizeof(T) * Trials);
    state.SetItemsProcessed(state.iterations() * size * Trials);

    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));
}

// IPT - items per thread
#define CREATE_BENCHMARK(T, BS, IPT) \
    benchmark::RegisterBenchmark( \
        (std::string("block_histogram<"#T", "#BS", "#IPT", " + algorithm_name + ">.") + method_name).c_str(), \
        run_benchmark<Benchmark, T, BS, IPT>, \
        stream, size \
    )

#define BENCHMARK_TYPE(type, block) \
    CREATE_BENCHMARK(type, block, 1), \
    CREATE_BENCHMARK(type, block, 2), \
    CREATE_BENCHMARK(type, block, 3), \
    CREATE_BENCHMARK(type, block, 4), \
    CREATE_BENCHMARK(type, block, 8), \
    CREATE_BENCHMARK(type, block, 16)

template<class Benchmark>
void add_benchmarks(std::vector<benchmark::internal::Benchmark*>& benchmarks,
                    const std::string& method_name,
                    const std::string& algorithm_name,
                    hipStream_t stream,
                    size_t size)
{
    std::vector<benchmark::internal::Benchmark*> new_benchmarks =
    {
        BENCHMARK_TYPE(int, 256),
        BENCHMARK_TYPE(int, 320),
        BENCHMARK_TYPE(int, 512),

        BENCHMARK_TYPE(unsigned long long, 256),
        BENCHMARK_TYPE(unsigned long long, 320)
    };
    benchmarks.insert(benchmarks.end(), new_benchmarks.begin(), new_benchmarks.end());
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
    // using_atomic
    using histogram_a_t = histogram<rocprim::block_histogram_algorithm::using_atomic>;
    add_benchmarks<histogram_a_t>(
        benchmarks, "histogram", "using_atomic", stream, size
    );
    // using_sort
    using histogram_s_t = histogram<rocprim::block_histogram_algorithm::using_sort>;
    add_benchmarks<histogram_s_t>(
        benchmarks, "histogram", "using_sort", stream, size
    );

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
