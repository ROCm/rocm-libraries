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
#include <hip/hip_hcc.h>

// rocPRIM
#include <rocprim.hpp>

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

struct reduce
{
    template<
        class T,
        unsigned int WarpSize,
        unsigned int Trials
    >
    __global__
    static void kernel(const T * d_input, T * d_output)
    {
        const unsigned int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

        auto value = d_input[i];

        using wreduce_t = rp::warp_reduce<T, WarpSize>;
        __shared__ typename wreduce_t::storage_type storage;
        #pragma nounroll
        for(unsigned int trial = 0; trial < Trials; trial++)
        {
            wreduce_t().reduce(value, value, storage);
        }

        d_output[i] = value;
    }
};

struct all_reduce
{
    template<
        class T,
        unsigned int WarpSize,
        unsigned int Trials
    >
    __global__
    static void kernel(const T * d_input, T * d_output)
    {
        const unsigned int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

        auto value = d_input[i];

        using wreduce_t = rp::warp_reduce<T, WarpSize, true>;
        __shared__ typename wreduce_t::storage_type storage;
        #pragma nounroll
        for(unsigned int trial = 0; trial < Trials; trial++)
        {
            wreduce_t().reduce(value, value, storage);
        }

        d_output[i] = value;
    }
};

template<
    class Benchmark,
    class T,
    unsigned int WarpSize,
    unsigned int BlockSize,
    unsigned int Trials = 100
>
void run_benchmark(benchmark::State& state, hipStream_t stream, size_t N)
{
    const auto size = BlockSize * ((N + BlockSize - 1)/BlockSize);

    std::vector<T> input = get_random_data<T>(size, T(0), T(10));
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

        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(Benchmark::template kernel<T, WarpSize, Trials>),
            dim3(size/BlockSize), dim3(BlockSize), 0, stream,
            d_input, d_output
        );
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

#define CREATE_BENCHMARK(T, WS, BS) \
benchmark::RegisterBenchmark( \
    (std::string("warp_reduce<" #T ", " #WS ", " #BS ">.") + name).c_str(), \
    run_benchmark<Benchmark, T, WS, BS>, \
    stream, size \
)

template<class Benchmark>
void add_benchmarks(const std::string& name,
                    std::vector<benchmark::internal::Benchmark*>& benchmarks,
                    hipStream_t stream,
                    size_t size)
{
    std::vector<benchmark::internal::Benchmark*> bs =
    {
        CREATE_BENCHMARK(int, 32, 64),
        CREATE_BENCHMARK(int, 64, 64),
        CREATE_BENCHMARK(int, 37, 64),
        CREATE_BENCHMARK(int, 61, 64),
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
    add_benchmarks<reduce>("reduce", benchmarks, stream, size);
    add_benchmarks<all_reduce>("all_reduce", benchmarks, stream, size);

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
