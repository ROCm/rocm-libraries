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

#include <iostream>

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

#ifndef DEFAULT_N
const size_t DEFAULT_N = 1024 * 1024 * 32;
#endif

template<
    bool AllReduce,
    class T,
    unsigned int WarpSize,
    unsigned int Trials
>
__global__
__launch_bounds__(ROCPRIM_DEFAULT_MAX_BLOCK_SIZE)
void warp_reduce_kernel(const T * d_input, T * d_output)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    auto value = d_input[i];

    using wreduce_t = rocprim::warp_reduce<T, WarpSize, AllReduce>;
    __shared__ typename wreduce_t::storage_type storage;
    ROCPRIM_NO_UNROLL
    for(unsigned int trial = 0; trial < Trials; trial++)
    {
        wreduce_t().reduce(value, value, storage);
    }

    d_output[i] = value;
}

template<
    class T,
    class Flag,
    unsigned int WarpSize,
    unsigned int Trials
>
__global__
__launch_bounds__(ROCPRIM_DEFAULT_MAX_BLOCK_SIZE)
void segmented_warp_reduce_kernel(const T* d_input, Flag* d_flags, T* d_output)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    auto value = d_input[i];
    auto flag = d_flags[i];

    using wreduce_t = rocprim::warp_reduce<T, WarpSize>;
    __shared__ typename wreduce_t::storage_type storage;
    ROCPRIM_NO_UNROLL
    for(unsigned int trial = 0; trial < Trials; trial++)
    {
        wreduce_t().head_segmented_reduce(value, value, flag, storage);
    }

    d_output[i] = value;
}

template<
    bool AllReduce,
    bool Segmented,
    unsigned int WarpSize,
    unsigned int BlockSize,
    unsigned int Trials,
    class T,
    class Flag
>
inline
auto execute_warp_reduce_kernel(T* input, T* output, Flag* /* flags */,
                                size_t size, hipStream_t stream)
    -> typename std::enable_if<!Segmented>::type
{
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(warp_reduce_kernel<AllReduce, T, WarpSize, Trials>),
        dim3(size/BlockSize), dim3(BlockSize), 0, stream,
        input, output
    );
    HIP_CHECK(hipGetLastError());
}

template<
    bool AllReduce,
    bool Segmented,
    unsigned int WarpSize,
    unsigned int BlockSize,
    unsigned int Trials,
    class T,
    class Flag
>
inline
auto execute_warp_reduce_kernel(T* input, T* output, Flag* flags,
                                size_t size, hipStream_t stream)
    -> typename std::enable_if<Segmented>::type
{
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(segmented_warp_reduce_kernel<T, Flag, WarpSize, Trials>),
        dim3(size/BlockSize), dim3(BlockSize), 0, stream,
        input, flags, output
    );
    HIP_CHECK(hipGetLastError());
}

template<
    bool AllReduce,
    bool Segmented,
    class T,
    unsigned int WarpSize,
    unsigned int BlockSize,
    unsigned int Trials = 100
>
void run_benchmark(benchmark::State& state, hipStream_t stream, size_t N)
{
    using flag_type = unsigned char;

    const auto size = BlockSize * ((N + BlockSize - 1)/BlockSize);

    std::vector<T> input = get_random_data<T>(size, T(0), T(10));
    std::vector<flag_type> flags = get_random_data<flag_type>(size, 0, 1);
    T * d_input;
    flag_type * d_flags;
    T * d_output;
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_input), size * sizeof(T)));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_flags), size * sizeof(flag_type)));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_output), size * sizeof(T)));
    HIP_CHECK(
        hipMemcpy(
            d_input, input.data(),
            size * sizeof(T),
            hipMemcpyHostToDevice
        )
    );
    HIP_CHECK(
        hipMemcpy(
            d_flags, flags.data(),
            size * sizeof(flag_type),
            hipMemcpyHostToDevice
        )
    );
    HIP_CHECK(hipDeviceSynchronize());

    // HIP events creation
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    for(auto _ : state)
    {
        // Record start event
        HIP_CHECK(hipEventRecord(start, stream));

        execute_warp_reduce_kernel<AllReduce, Segmented, WarpSize, BlockSize, Trials>(d_input,
                                                                                      d_output,
                                                                                      d_flags,
                                                                                      size,
                                                                                      stream);

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

    state.SetBytesProcessed(state.iterations() * Trials * size * sizeof(T));
    state.SetItemsProcessed(state.iterations() * Trials * size);

    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));
    HIP_CHECK(hipFree(d_flags));
}

#define CREATE_BENCHMARK(T, WS, BS)                                                           \
    benchmark::RegisterBenchmark(                                                             \
        bench_naming::format_name("{lvl:warp,algo:reduce,key_type:" #T ",broadcast_result:"   \
                                  + std::string(AllReduce ? "true" : "false")                 \
                                  + ",segmented:" + std::string(Segmented ? "true" : "false") \
                                  + ",ws:" #WS ",cfg:{bs:" #BS "}}")                          \
            .c_str(),                                                                         \
        run_benchmark<AllReduce, Segmented, T, WS, BS>,                                       \
        stream,                                                                               \
        size)

#define BENCHMARK_TYPE(type) \
    CREATE_BENCHMARK(type, 32, 64), \
    CREATE_BENCHMARK(type, 37, 64), \
    CREATE_BENCHMARK(type, 61, 64), \
    CREATE_BENCHMARK(type, 64, 64)

template<bool AllReduce, bool Segmented>
void add_benchmarks(std::vector<benchmark::internal::Benchmark*>& benchmarks,
                    hipStream_t                                   stream,
                    size_t                                        size)
{
    std::vector<benchmark::internal::Benchmark*> bs =
    {
        BENCHMARK_TYPE(int),
        BENCHMARK_TYPE(float),
        BENCHMARK_TYPE(double),
        BENCHMARK_TYPE(int8_t),
        BENCHMARK_TYPE(uint8_t),
        BENCHMARK_TYPE(rocprim::half)
    };

    benchmarks.insert(benchmarks.end(), bs.begin(), bs.end());
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
    add_benchmarks<false, false>(benchmarks, stream, size);
    add_benchmarks<true, false>(benchmarks, stream, size);
    add_benchmarks<false, true>(benchmarks, stream, size);

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
