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

namespace rp = rocprim;

template<
    class Runner,
    class T,
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    bool WithTile,
    unsigned int Trials
>
__global__
void kernel(const T * d_input, T * d_output)
{
    Runner::template run<T, BlockSize, ItemsPerThread, WithTile, Trials>(d_input, d_output);
}

struct flag_heads
{
    template<
        class T,
        unsigned int BlockSize,
        unsigned int ItemsPerThread,
        bool WithTile,
        unsigned int Trials
    >
    __device__
    static void run(const T * d_input, T * d_output)
    {
        const unsigned int lid = hipThreadIdx_x;
        const unsigned int block_offset = hipBlockIdx_x * ItemsPerThread * BlockSize;

        T input[ItemsPerThread];
        rp::block_load_direct_striped<BlockSize>(lid, d_input + block_offset, input);

        #pragma nounroll
        for(unsigned int trial = 0; trial < Trials; trial++)
        {
            rp::block_discontinuity<T, BlockSize> bdiscontinuity;
            bool head_flags[ItemsPerThread];
            if(WithTile)
            {
                bdiscontinuity.flag_heads(head_flags, T(123), input, rp::equal_to<T>());
            }
            else
            {
                bdiscontinuity.flag_heads(head_flags, input, rp::equal_to<T>());
            }

            for(unsigned int i = 0; i < ItemsPerThread; i++)
            {
                input[i] += head_flags[i];
            }
            rp::syncthreads();
        }

        rp::block_store_direct_striped<BlockSize>(lid, d_output + block_offset, input);
    }
};

struct flag_tails
{
    template<
        class T,
        unsigned int BlockSize,
        unsigned int ItemsPerThread,
        bool WithTile,
        unsigned int Trials
    >
    __device__
    static void run(const T * d_input, T * d_output)
    {
        const unsigned int lid = hipThreadIdx_x;
        const unsigned int block_offset = hipBlockIdx_x * ItemsPerThread * BlockSize;

        T input[ItemsPerThread];
        rp::block_load_direct_striped<BlockSize>(lid, d_input + block_offset, input);

        #pragma nounroll
        for(unsigned int trial = 0; trial < Trials; trial++)
        {
            rp::block_discontinuity<T, BlockSize> bdiscontinuity;
            bool tail_flags[ItemsPerThread];
            if(WithTile)
            {
                bdiscontinuity.flag_tails(tail_flags, T(123), input, rp::equal_to<T>());
            }
            else
            {
                bdiscontinuity.flag_tails(tail_flags, input, rp::equal_to<T>());
            }

            for(unsigned int i = 0; i < ItemsPerThread; i++)
            {
                input[i] += tail_flags[i];
            }
            rp::syncthreads();
        }

        rp::block_store_direct_striped<BlockSize>(lid, d_output + block_offset, input);
    }
};

struct flag_heads_and_tails
{
    template<
        class T,
        unsigned int BlockSize,
        unsigned int ItemsPerThread,
        bool WithTile,
        unsigned int Trials
    >
    __device__
    static void run(const T * d_input, T * d_output)
    {
        const unsigned int lid = hipThreadIdx_x;
        const unsigned int block_offset = hipBlockIdx_x * ItemsPerThread * BlockSize;

        T input[ItemsPerThread];
        rp::block_load_direct_striped<BlockSize>(lid, d_input + block_offset, input);

        #pragma nounroll
        for(unsigned int trial = 0; trial < Trials; trial++)
        {
            rp::block_discontinuity<T, BlockSize> bdiscontinuity;
            bool head_flags[ItemsPerThread];
            bool tail_flags[ItemsPerThread];
            if(WithTile)
            {
                bdiscontinuity.flag_heads_and_tails(head_flags, T(123), tail_flags, T(234), input, rp::equal_to<T>());
            }
            else
            {
                bdiscontinuity.flag_heads_and_tails(head_flags, tail_flags, input, rp::equal_to<T>());
            }

            for(unsigned int i = 0; i < ItemsPerThread; i++)
            {
                input[i] += head_flags[i];
                input[i] += tail_flags[i];
            }
            rp::syncthreads();
        }

        rp::block_store_direct_striped<BlockSize>(lid, d_output + block_offset, input);
    }
};

template<
    class Benchmark,
    class T,
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    bool WithTile,
    unsigned int Trials = 100
>
void run_benchmark(benchmark::State& state, hipStream_t stream, size_t N)
{
    constexpr auto items_per_block = BlockSize * ItemsPerThread;
    const auto size = items_per_block * ((N + items_per_block - 1)/items_per_block);

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
            HIP_KERNEL_NAME(kernel<Benchmark, T, BlockSize, ItemsPerThread, WithTile, Trials>),
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
    state.SetBytesProcessed(state.iterations() * Trials * size * sizeof(T));
    state.SetItemsProcessed(state.iterations() * Trials * size);

    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));
}

#define CREATE_BENCHMARK(T, BS, IPT, WITH_TILE) \
benchmark::RegisterBenchmark( \
    (std::string("block_discontinuity<" #T ", " #BS ">.") + name + ("<" #IPT ", " #WITH_TILE ">")).c_str(), \
    run_benchmark<Benchmark, T, BS, IPT, WITH_TILE>, \
    stream, size \
)

#define BENCHMARK_TYPE(type, block, bool) \
    CREATE_BENCHMARK(type, block, 1, bool), \
    CREATE_BENCHMARK(type, block, 2, bool), \
    CREATE_BENCHMARK(type, block, 3, bool), \
    CREATE_BENCHMARK(type, block, 4, bool), \
    CREATE_BENCHMARK(type, block, 8, bool)


template<class Benchmark>
void add_benchmarks(const std::string& name,
                    std::vector<benchmark::internal::Benchmark*>& benchmarks,
                    hipStream_t stream,
                    size_t size)
{
    std::vector<benchmark::internal::Benchmark*> bs =
    {
        BENCHMARK_TYPE(int, 256, false),
        BENCHMARK_TYPE(int, 256, true),
        BENCHMARK_TYPE(int8_t, 256, false),
        BENCHMARK_TYPE(int8_t, 256, true),
        BENCHMARK_TYPE(uint8_t, 256, false),
        BENCHMARK_TYPE(uint8_t, 256, true),
        BENCHMARK_TYPE(rocprim::half, 256, false),
        BENCHMARK_TYPE(rocprim::half, 256, true),
        BENCHMARK_TYPE(long long, 256, false),
        BENCHMARK_TYPE(long long, 256, true),
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
    add_benchmarks<flag_heads>("flag_heads", benchmarks, stream, size);
    add_benchmarks<flag_tails>("flag_tails", benchmarks, stream, size);
    add_benchmarks<flag_heads_and_tails>("flag_heads_and_tails", benchmarks, stream, size);

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
