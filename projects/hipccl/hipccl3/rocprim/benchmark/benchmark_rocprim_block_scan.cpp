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
#include <string>
#include <cstdio>

// Google Benchmark
#include "benchmark/benchmark.h"

// CmdParser
#include "cmdparser.hpp"

// HC API
#include <hcc/hc.hpp>
// HIP API
#include <hip/hip_runtime.h>
#include <hip/hip_hcc.h>

// rocPRIM
#include <block/block_scan.hpp>

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

template<rocprim::block_scan_algorithm algorithm>
struct inclusive_scan
{
    template<
        class T,
        unsigned int BlockSize,
        unsigned int ItemsPerThread,
        unsigned int Trials
    >
    __global__
    static void kernel(const T* input, T* output)
    {
        const unsigned int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

        T values[ItemsPerThread];
        for(unsigned int k = 0; k < ItemsPerThread; k++)
        {
            values[k] = input[i * ItemsPerThread + k];
        }

        using bscan_t = rp::block_scan<T, BlockSize, algorithm>;
        __shared__ typename bscan_t::storage_type storage;

        #pragma nounroll
        for(unsigned int trial = 0; trial < Trials; trial++)
        {
            bscan_t().inclusive_scan(values, values, storage);
        }

        for(unsigned int k = 0; k < ItemsPerThread; k++)
        {
            output[i * ItemsPerThread + k] = values[k];
        }
    }
};

template<
    class Benchmark,
    class T,
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    unsigned int Trials = 100
>
void run_benchmark(benchmark::State& state, hipStream_t stream, size_t N)
{
    // Make sure size is a multiple of BlockSize
    constexpr auto items_per_block = BlockSize * ItemsPerThread;
    const auto size = items_per_block * ((N + items_per_block - 1)/items_per_block);
    // Allocate and fill memory
    std::vector<T> input(size, 1.0f);
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

    for (auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(Benchmark::template kernel<T, BlockSize, ItemsPerThread, Trials>),
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

std::string create_benchmark_name(std::string name_template,
                                  const unsigned int ipt)
{
    std::vector<char> buffer(name_template.size() + 100);
    std::snprintf(buffer.data(), buffer.size(), name_template.c_str(), ipt);
    std::string result = buffer.data();
    return result;
}

// IPT - items per thread
#define CREATE_BENCHMARK(IPT) \
    benchmark::RegisterBenchmark( \
        create_benchmark_name(benchmark_name_template, IPT).c_str(), \
        run_benchmark<Benchmark, T, BlockSize, IPT>, \
        stream, size \
    )

template<
    class Benchmark,
    class T,
    unsigned int BlockSize
>
void add_benchmarks(std::vector<benchmark::internal::Benchmark*>& benchmarks,
                    const std::string& benchmark_name_template,
                    hipStream_t stream,
                    size_t size)
{
    std::vector<benchmark::internal::Benchmark*> new_benchmarks =
    {
        CREATE_BENCHMARK(1),
        CREATE_BENCHMARK(2),
        CREATE_BENCHMARK(3),
        CREATE_BENCHMARK(4),
        CREATE_BENCHMARK(8),
        CREATE_BENCHMARK(16)
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

    // using_warp_scan
    using inclusive_scan_uws_t = inclusive_scan<rocprim::block_scan_algorithm::using_warp_scan>;
    add_benchmarks<inclusive_scan_uws_t, float, 256>(
        benchmarks, "block_scan.inclusive_scan<float, 256, %u>", stream, size
    );
    add_benchmarks<inclusive_scan_uws_t, int, 256>(
        benchmarks, "block_scan.inclusive_scan<int, 256, %u>", stream, size
    );
    add_benchmarks<inclusive_scan_uws_t, double, 256>(
        benchmarks, "block_scan.inclusive_scan<double, 256, %u>", stream, size
    );

    // reduce then scan
    using inclusive_scan_rts_t = inclusive_scan<rocprim::block_scan_algorithm::reduce_then_scan>;
    add_benchmarks<inclusive_scan_rts_t, float, 256>(
        benchmarks, "block_scan.inclusive_scan<float, 256, %u>", stream, size
    );
    add_benchmarks<inclusive_scan_rts_t, int, 256>(
        benchmarks, "block_scan.inclusive_scan<int, 256, %u>", stream, size
    );
    add_benchmarks<inclusive_scan_rts_t, double, 256>(
        benchmarks, "block_scan.inclusive_scan<double, 256, %u>", stream, size
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
