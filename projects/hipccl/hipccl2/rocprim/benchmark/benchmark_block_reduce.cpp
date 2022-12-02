// MIT License
//
// Copyright (c) 2017-2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

// Google Benchmark
#include "benchmark/benchmark.h"
// CmdParser
#include "cmdparser.hpp"
#include "benchmark_utils.hpp"

// HIP API
#include <hip/hip_runtime.h>

// rocPRIM
#include <rocprim/rocprim.hpp>

#ifndef DEFAULT_N
const size_t DEFAULT_N = 1024 * 1024 * 32;
#endif

namespace rp = rocprim;

template<
    class Runner,
    class T,
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    unsigned int Trials
>
__global__
__launch_bounds__(BlockSize)
void kernel(const T* input, T* output)
{
    Runner::template run<T, BlockSize, ItemsPerThread, Trials>(input, output);
}

template<rocprim::block_reduce_algorithm algorithm>
struct reduce
{
    template<
        class T,
        unsigned int BlockSize,
        unsigned int ItemsPerThread,
        unsigned int Trials
    >
    __device__
    static void run(const T* input, T* output)
    {
        const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

        T values[ItemsPerThread];
        T reduced_value;
        for(unsigned int k = 0; k < ItemsPerThread; k++)
        {
            values[k] = input[i * ItemsPerThread + k];
        }

        using breduce_t = rp::block_reduce<T, BlockSize, algorithm>;
        __shared__ typename breduce_t::storage_type storage;

        ROCPRIM_NO_UNROLL
        for(unsigned int trial = 0; trial < Trials; trial++)
        {
            breduce_t().reduce(values, reduced_value, storage);
            values[0] = reduced_value;
        }

        if(threadIdx.x == 0)
        {
            output[blockIdx.x] = reduced_value;
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
    std::vector<T> input(size, T(1));
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

    // HIP events creation
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    for (auto _ : state)
    {
        // Record start event
        HIP_CHECK(hipEventRecord(start, stream));

        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(kernel<Benchmark, T, BlockSize, ItemsPerThread, Trials>),
            dim3(size/items_per_block), dim3(BlockSize), 0, stream,
            d_input, d_output
        );
        HIP_CHECK(hipGetLastError());

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

    state.SetBytesProcessed(state.iterations() * size * sizeof(T) * Trials);
    state.SetItemsProcessed(state.iterations() * size * Trials);

    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));
}

// IPT - items per thread
#define CREATE_BENCHMARK(T, BS, IPT)                                                               \
    benchmark::RegisterBenchmark(bench_naming::format_name("{lvl:block,algo:reduce,key_type:" #T   \
                                                           ",cfg:{bs:" #BS ",ipt:" #IPT ",method:" \
                                                           + method_name + "}}")                   \
                                     .c_str(),                                                     \
                                 run_benchmark<Benchmark, T, BS, IPT>,                             \
                                 stream,                                                           \
                                 size)

#define BENCHMARK_TYPE(type, block) \
    CREATE_BENCHMARK(type, block, 1), \
    CREATE_BENCHMARK(type, block, 2), \
    CREATE_BENCHMARK(type, block, 3), \
    CREATE_BENCHMARK(type, block, 4), \
    CREATE_BENCHMARK(type, block, 8), \
    CREATE_BENCHMARK(type, block, 11), \
    CREATE_BENCHMARK(type, block, 16)

template<class Benchmark>
void add_benchmarks(std::vector<benchmark::internal::Benchmark*>& benchmarks,
                    const std::string&                            method_name,
                    hipStream_t                                   stream,
                    size_t                                        size)
{
    using custom_float2 = custom_type<float, float>;
    using custom_double2 = custom_type<double, double>;

    std::vector<benchmark::internal::Benchmark*> new_benchmarks =
    {
        // When block size is less than or equal to warp size
        BENCHMARK_TYPE(int, 64),
        BENCHMARK_TYPE(float, 64),
        BENCHMARK_TYPE(double, 64),
        BENCHMARK_TYPE(int8_t, 64),
        BENCHMARK_TYPE(uint8_t, 64),
        BENCHMARK_TYPE(rocprim::half, 64),

        BENCHMARK_TYPE(int, 256),
        BENCHMARK_TYPE(float, 256),
        BENCHMARK_TYPE(double, 256),
        BENCHMARK_TYPE(int8_t, 256),
        BENCHMARK_TYPE(uint8_t, 256),
        BENCHMARK_TYPE(rocprim::half, 256),

        CREATE_BENCHMARK(custom_float2, 256, 1),
        CREATE_BENCHMARK(custom_float2, 256, 4),
        CREATE_BENCHMARK(custom_float2, 256, 8),

        CREATE_BENCHMARK(float2, 256, 1),
        CREATE_BENCHMARK(float2, 256, 4),
        CREATE_BENCHMARK(float2, 256, 8),

        CREATE_BENCHMARK(custom_double2, 256, 1),
        CREATE_BENCHMARK(custom_double2, 256, 4),
        CREATE_BENCHMARK(custom_double2, 256, 8),

        CREATE_BENCHMARK(double2, 256, 1),
        CREATE_BENCHMARK(double2, 256, 4),
        CREATE_BENCHMARK(double2, 256, 8),

        CREATE_BENCHMARK(float4, 256, 1),
        CREATE_BENCHMARK(float4, 256, 4),
        CREATE_BENCHMARK(float4, 256, 8),
    };
    benchmarks.insert(benchmarks.end(), new_benchmarks.begin(), new_benchmarks.end());
}

int main(int argc, char *argv[])
{
    cli::Parser parser(argc, argv);
    parser.set_optional<size_t>("size", "size", DEFAULT_N, "number of values");
    parser.set_optional<int>("trials", "trials", -1, "number of iterations");
    parser.set_optional<std::string>("name_format",
                                     "name_format",
                                     "human",
                                     "either: json,human,txt");
    parser.run_and_exit_if_error();

    // Parse argv
    benchmark::Initialize(&argc, argv);
    const size_t size = parser.get<size_t>("size");
    const int trials = parser.get<int>("trials");
    bench_naming::set_format(parser.get<std::string>("name_format"));

    // HIP
    hipStream_t stream = 0; // default

    // Benchmark info
    add_common_benchmark_info();
    benchmark::AddCustomContext("size", std::to_string(size));

    // Add benchmarks
    std::vector<benchmark::internal::Benchmark*> benchmarks;
    // using_warp_scan
    using reduce_uwr_t = reduce<rocprim::block_reduce_algorithm::using_warp_reduce>;
    add_benchmarks<reduce_uwr_t>(benchmarks, "using_warp_reduce", stream, size);
    // reduce then scan
    using reduce_rr_t = reduce<rocprim::block_reduce_algorithm::raking_reduce>;
    add_benchmarks<reduce_rr_t>(benchmarks, "raking_reduce", stream, size);
    // reduce commutative only
    using reduce_rrco_t = reduce<rocprim::block_reduce_algorithm::raking_reduce_commutative_only>;
    add_benchmarks<reduce_rrco_t>(benchmarks, "raking_reduce_commutative_only", stream, size);

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
