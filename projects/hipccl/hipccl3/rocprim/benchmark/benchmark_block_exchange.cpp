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
void kernel(const T * d_input, const unsigned int * d_ranks, T * d_output)
{
    Runner::template run<T, BlockSize, ItemsPerThread, Trials>(d_input, d_ranks, d_output);
}

struct blocked_to_striped
{
    template<
        class T,
        unsigned int BlockSize,
        unsigned int ItemsPerThread,
        unsigned int Trials
    >
    __device__
    static void run(const T * d_input, const unsigned int *, T * d_output)
    {
        const unsigned int lid = threadIdx.x;
        const unsigned int block_offset = blockIdx.x * ItemsPerThread * BlockSize;

        T input[ItemsPerThread];
        rp::block_load_direct_striped<BlockSize>(lid, d_input + block_offset, input);

        ROCPRIM_NO_UNROLL
        for(unsigned int trial = 0; trial < Trials; trial++)
        {
            rp::block_exchange<T, BlockSize, ItemsPerThread> exchange;
            exchange.blocked_to_striped(input, input);
            ::rocprim::syncthreads();
        }

        rp::block_store_direct_striped<BlockSize>(lid, d_output + block_offset, input);
    }
};

struct striped_to_blocked
{
    template<
        class T,
        unsigned int BlockSize,
        unsigned int ItemsPerThread,
        unsigned int Trials
    >
    __device__
    static void run(const T * d_input, const unsigned int *, T * d_output)
    {
        const unsigned int lid = threadIdx.x;
        const unsigned int block_offset = blockIdx.x * ItemsPerThread * BlockSize;

        T input[ItemsPerThread];
        rp::block_load_direct_striped<BlockSize>(lid, d_input + block_offset, input);

        ROCPRIM_NO_UNROLL
        for(unsigned int trial = 0; trial < Trials; trial++)
        {
            rp::block_exchange<T, BlockSize, ItemsPerThread> exchange;
            exchange.striped_to_blocked(input, input);
            ::rocprim::syncthreads();
        }

        rp::block_store_direct_striped<BlockSize>(lid, d_output + block_offset, input);
    }
};

struct blocked_to_warp_striped
{
    template<
        class T,
        unsigned int BlockSize,
        unsigned int ItemsPerThread,
        unsigned int Trials
    >
    __device__
    static void run(const T * d_input, const unsigned int *, T * d_output)
    {
        const unsigned int lid = threadIdx.x;
        const unsigned int block_offset = blockIdx.x * ItemsPerThread * BlockSize;

        T input[ItemsPerThread];
        rp::block_load_direct_striped<BlockSize>(lid, d_input + block_offset, input);

        ROCPRIM_NO_UNROLL
        for(unsigned int trial = 0; trial < Trials; trial++)
        {
            rp::block_exchange<T, BlockSize, ItemsPerThread> exchange;
            exchange.blocked_to_warp_striped(input, input);
            ::rocprim::syncthreads();
        }

        rp::block_store_direct_striped<BlockSize>(lid, d_output + block_offset, input);
    }
};

struct warp_striped_to_blocked
{
    template<
        class T,
        unsigned int BlockSize,
        unsigned int ItemsPerThread,
        unsigned int Trials
    >
    __device__
    static void run(const T * d_input, const unsigned int *, T * d_output)
    {
        const unsigned int lid = threadIdx.x;
        const unsigned int block_offset = blockIdx.x * ItemsPerThread * BlockSize;

        T input[ItemsPerThread];
        rp::block_load_direct_striped<BlockSize>(lid, d_input + block_offset, input);

        ROCPRIM_NO_UNROLL
        for(unsigned int trial = 0; trial < Trials; trial++)
        {
            rp::block_exchange<T, BlockSize, ItemsPerThread> exchange;
            exchange.warp_striped_to_blocked(input, input);
            ::rocprim::syncthreads();
        }

        rp::block_store_direct_striped<BlockSize>(lid, d_output + block_offset, input);
    }
};

struct scatter_to_blocked
{
    template<
        class T,
        unsigned int BlockSize,
        unsigned int ItemsPerThread,
        unsigned int Trials
    >
    __device__
    static void run(const T * d_input, const unsigned int * d_ranks, T * d_output)
    {
        const unsigned int lid = threadIdx.x;
        const unsigned int block_offset = blockIdx.x * ItemsPerThread * BlockSize;

        T input[ItemsPerThread];
        unsigned int ranks[ItemsPerThread];
        rp::block_load_direct_striped<BlockSize>(lid, d_input + block_offset, input);
        rp::block_load_direct_striped<BlockSize>(lid, d_ranks + block_offset, ranks);

        ROCPRIM_NO_UNROLL
        for(unsigned int trial = 0; trial < Trials; trial++)
        {
            rp::block_exchange<T, BlockSize, ItemsPerThread> exchange;
            exchange.scatter_to_blocked(input, input, ranks);
            ::rocprim::syncthreads();
        }

        rp::block_store_direct_striped<BlockSize>(lid, d_output + block_offset, input);
    }
};

struct scatter_to_striped
{
    template<
        class T,
        unsigned int BlockSize,
        unsigned int ItemsPerThread,
        unsigned int Trials
    >
    __device__
    static void run(const T * d_input, const unsigned int * d_ranks, T * d_output)
    {
        const unsigned int lid = threadIdx.x;
        const unsigned int block_offset = blockIdx.x * ItemsPerThread * BlockSize;

        T input[ItemsPerThread];
        unsigned int ranks[ItemsPerThread];
        rp::block_load_direct_striped<BlockSize>(lid, d_input + block_offset, input);
        rp::block_load_direct_striped<BlockSize>(lid, d_ranks + block_offset, ranks);

        ROCPRIM_NO_UNROLL
        for(unsigned int trial = 0; trial < Trials; trial++)
        {
            rp::block_exchange<T, BlockSize, ItemsPerThread> exchange;
            exchange.scatter_to_striped(input, input, ranks);
            ::rocprim::syncthreads();
        }

        rp::block_store_direct_striped<BlockSize>(lid, d_output + block_offset, input);
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
    constexpr auto items_per_block = BlockSize * ItemsPerThread;
    const auto size = items_per_block * ((N + items_per_block - 1)/items_per_block);

    std::vector<T> input(size);
    // Fill input
    for(size_t i = 0; i < size; i++)
    {
        input[i] = T(i);
    }
    std::vector<unsigned int> ranks(size);
    // Fill ranks (for scatter operations)
    std::mt19937 gen;
    for(size_t bi = 0; bi < size / items_per_block; bi++)
    {
        auto block_ranks = ranks.begin() + bi * items_per_block;
        std::iota(block_ranks, block_ranks + items_per_block, 0);
        std::shuffle(block_ranks, block_ranks + items_per_block, gen);
    }
    T * d_input;
    unsigned int * d_ranks;
    T * d_output;
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_input), size * sizeof(T)));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_ranks), size * sizeof(unsigned int)));
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
            d_ranks, ranks.data(),
            size * sizeof(unsigned int),
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

        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(kernel<Benchmark, T, BlockSize, ItemsPerThread, Trials>),
            dim3(size/items_per_block), dim3(BlockSize), 0, stream,
            d_input, d_ranks, d_output
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

    state.SetBytesProcessed(state.iterations() * Trials * size * sizeof(T));
    state.SetItemsProcessed(state.iterations() * Trials * size);

    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_ranks));
    HIP_CHECK(hipFree(d_output));
}

#define CREATE_BENCHMARK(T, BS, IPT)                                                   \
    benchmark::RegisterBenchmark(                                                      \
        bench_naming::format_name("{lvl:block,algo:exchange,subalgo:" + name           \
                                  + ",key_type:" #T ",cfg:{bs:" #BS ",ipt:" #IPT "}}") \
            .c_str(),                                                                  \
        run_benchmark<Benchmark, T, BS, IPT>,                                          \
        stream,                                                                        \
        size)

#define BENCHMARK_TYPE(type, block) \
    CREATE_BENCHMARK(type, block, 1), \
    CREATE_BENCHMARK(type, block, 2), \
    CREATE_BENCHMARK(type, block, 3), \
    CREATE_BENCHMARK(type, block, 4), \
    CREATE_BENCHMARK(type, block, 7), \
    CREATE_BENCHMARK(type, block, 8)

template<class Benchmark>
void add_benchmarks(const std::string& name,
                    std::vector<benchmark::internal::Benchmark*>& benchmarks,
                    hipStream_t stream,
                    size_t size)
{
    using custom_float2 = custom_type<float, float>;
    using custom_double2 = custom_type<double, double>;

    std::vector<benchmark::internal::Benchmark*> bs =
    {
        BENCHMARK_TYPE(int, 256),
        BENCHMARK_TYPE(int8_t, 256),
        BENCHMARK_TYPE(rocprim::half, 256),
        BENCHMARK_TYPE(long long, 256),
        BENCHMARK_TYPE(custom_float2, 256),
        BENCHMARK_TYPE(float2, 256),
        BENCHMARK_TYPE(custom_double2, 256),
        BENCHMARK_TYPE(double2, 256),
        BENCHMARK_TYPE(float4, 256),
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
    add_benchmarks<blocked_to_striped>("blocked_to_striped", benchmarks, stream, size);
    add_benchmarks<striped_to_blocked>("striped_to_blocked", benchmarks, stream, size);
    add_benchmarks<blocked_to_warp_striped>("blocked_to_warp_striped", benchmarks, stream, size);
    add_benchmarks<warp_striped_to_blocked>("warp_striped_to_blocked", benchmarks, stream, size);
    add_benchmarks<scatter_to_blocked>("scatter_to_blocked", benchmarks, stream, size);
    add_benchmarks<scatter_to_striped>("scatter_to_striped", benchmarks, stream, size);

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
