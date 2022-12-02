// MIT License
//
// Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include <algorithm>
#include <rocprim/block/block_adjacent_difference.hpp>

// Google Benchmark
#include "benchmark/benchmark.h"

// CmdParser
#include "cmdparser.hpp"
#include "benchmark_utils.hpp"

// HIP API
#include <hip/hip_runtime.h>

#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <cstdio>
#include <cstdlib>

#ifndef DEFAULT_N
const size_t DEFAULT_N = 1024 * 1024 * 128;
#endif

namespace rp = rocprim;

template <class Benchmark,
          unsigned int BlockSize,
          unsigned int ItemsPerThread,
          bool         WithTile,
          typename... Args>
__global__ __launch_bounds__(BlockSize) void kernel(Args ...args)
{
    Benchmark::template run<BlockSize, ItemsPerThread, WithTile>(args...);
}

struct subtract_left
{
    template <unsigned int BlockSize, unsigned int ItemsPerThread, bool WithTile, typename T>
    __device__ static void run(const T* d_input, T* d_output, unsigned int trials)
    {
        const unsigned int lid = threadIdx.x;
        const unsigned int block_offset = blockIdx.x * ItemsPerThread * BlockSize;

        T input[ItemsPerThread];
        rp::block_load_direct_striped<BlockSize>(lid, d_input + block_offset, input);

        using adjacent_diff_t = rp::block_adjacent_difference<T, BlockSize>;
        __shared__ typename adjacent_diff_t::storage_type storage;

        ROCPRIM_NO_UNROLL
        for(unsigned int trial = 0; trial < trials; trial++)
        {
            T output[ItemsPerThread];
            if(WithTile)
            {
                adjacent_diff_t().subtract_left(input, output, rp::minus<>{}, T(123), storage);
            }
            else
            {
                adjacent_diff_t().subtract_left(input, output, rp::minus<>{}, storage);
            }

            for(unsigned int i = 0; i < ItemsPerThread; ++i)
            {
                input[i] += output[i];
            }
            rp::syncthreads();
        }

        rp::block_store_direct_striped<BlockSize>(lid, d_output + block_offset, input);
    }
};

struct subtract_left_partial
{
    template <unsigned int BlockSize, unsigned int ItemsPerThread, bool WithTile, typename T>
    __device__ static void
        run(const T* d_input, const unsigned int* tile_sizes, T* d_output, unsigned int trials)
    {
        const unsigned int lid = threadIdx.x;
        const unsigned int block_offset = blockIdx.x * ItemsPerThread * BlockSize;

        T input[ItemsPerThread];
        rp::block_load_direct_striped<BlockSize>(lid, d_input + block_offset, input);

        using adjacent_diff_t = rp::block_adjacent_difference<T, BlockSize>;
        __shared__ typename adjacent_diff_t::storage_type storage;

        unsigned int tile_size = tile_sizes[blockIdx.x];

        // Try to evenly distribute the length of tile_sizes between all the trials
        const auto tile_size_diff = (BlockSize * ItemsPerThread) / trials + 1;

        ROCPRIM_NO_UNROLL
        for(unsigned int trial = 0; trial < trials; trial++)
        {
            T output[ItemsPerThread];
            if(WithTile)
            {
                adjacent_diff_t().subtract_left_partial(input, output, rp::minus<>{}, T(123), tile_size, storage);
            }
            else
            {
                adjacent_diff_t().subtract_left_partial(input, output, rp::minus<>{}, tile_size, storage);
            }

            for(unsigned int i = 0; i < ItemsPerThread; ++i)
            {
                input[i] += output[i];
            }

            // Change the tile_size to even out the distribution
            tile_size = (tile_size + tile_size_diff) % (BlockSize * ItemsPerThread);
            rp::syncthreads();
        }
        rp::block_store_direct_striped<BlockSize>(lid, d_output + block_offset, input);
    }
};

struct subtract_right
{
    template <unsigned int BlockSize,
              unsigned int ItemsPerThread,
              bool         WithTile,
              typename T>
    __device__ static void run(const T* d_input, T* d_output, unsigned int trials)
    {
        const unsigned int lid = threadIdx.x;
        const unsigned int block_offset = blockIdx.x * ItemsPerThread * BlockSize;

        T input[ItemsPerThread];
        rp::block_load_direct_striped<BlockSize>(lid, d_input + block_offset, input);

        using adjacent_diff_t = rp::block_adjacent_difference<T, BlockSize>;
        __shared__ typename adjacent_diff_t::storage_type storage;

        ROCPRIM_NO_UNROLL
        for(unsigned int trial = 0; trial < trials; trial++)
        {
            T output[ItemsPerThread];
            if(WithTile)
            {
                adjacent_diff_t().subtract_right(input, output, rp::minus<>{}, T(123), storage);
            }
            else
            {
                adjacent_diff_t().subtract_right(input, output, rp::minus<>{}, storage);
            }

            for(unsigned int i = 0; i < ItemsPerThread; ++i)
            {
                input[i] += output[i];
            }
            rp::syncthreads();
        }

        rp::block_store_direct_striped<BlockSize>(lid, d_output + block_offset, input);
    }
};

struct subtract_right_partial
{
    template <unsigned int BlockSize, unsigned int ItemsPerThread, bool WithTile, typename T>
    __device__ static void
        run(const T* d_input, const unsigned int* tile_sizes, T* d_output, unsigned int trials)
    {
        const unsigned int lid = threadIdx.x;
        const unsigned int block_offset = blockIdx.x * ItemsPerThread * BlockSize;

        T input[ItemsPerThread];
        rp::block_load_direct_striped<BlockSize>(lid, d_input + block_offset, input);

        using adjacent_diff_t = rp::block_adjacent_difference<T, BlockSize>;
        __shared__ typename adjacent_diff_t::storage_type storage;

        unsigned int tile_size = tile_sizes[blockIdx.x];
        // Try to evenly distribute the length of tile_sizes between all the trials
        const auto tile_size_diff = (BlockSize * ItemsPerThread) / trials + 1;

        ROCPRIM_NO_UNROLL
        for(unsigned int trial = 0; trial < trials; trial++)
        {
            T output[ItemsPerThread];
            adjacent_diff_t().subtract_right_partial(input, output, rp::minus<>{}, tile_size, storage);

            for(unsigned int i = 0; i < ItemsPerThread; ++i)
            {
                input[i] += output[i];
            }
            // Change the tile_size to even out the distribution
            tile_size = (tile_size + tile_size_diff) % (BlockSize * ItemsPerThread);
            rp::syncthreads();
        }
        rp::block_store_direct_striped<BlockSize>(lid, d_output + block_offset, input);
    }
};

template <class Benchmark,
          class T,
          unsigned int BlockSize,
          unsigned int ItemsPerThread,
          bool         WithTile,
          unsigned int Trials = 100>
auto run_benchmark(benchmark::State& state, hipStream_t stream, size_t N)
    -> std::enable_if_t<!std::is_same<Benchmark, subtract_left_partial>::value
                        && !std::is_same<Benchmark, subtract_right_partial>::value>
{
    constexpr auto items_per_block = BlockSize * ItemsPerThread;
    const auto num_blocks = (N + items_per_block - 1) / items_per_block;
    // Round up size to the next multiple of items_per_block
    const auto size = num_blocks * items_per_block;

    const std::vector<T> input = get_random_data<T>(size, T(0), T(10));
    T* d_input;
    T* d_output;
    HIP_CHECK(hipMalloc(&d_input, input.size() * sizeof(input[0])));
    HIP_CHECK(hipMalloc(&d_output, input.size() * sizeof(T)));
    HIP_CHECK(
        hipMemcpy(
            d_input, input.data(),
            input.size() * sizeof(input[0]),
            hipMemcpyHostToDevice
        )
    );

    // HIP events creation
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    for(auto _ : state)
    {
        // Record start event
        HIP_CHECK(hipEventRecord(start, stream));

        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(kernel<Benchmark, BlockSize, ItemsPerThread, WithTile>),
            dim3(num_blocks), dim3(BlockSize), 0, stream,
            d_input, d_output, Trials
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
    HIP_CHECK(hipFree(d_output));
}

template <class Benchmark,
          class T,
          unsigned int BlockSize,
          unsigned int ItemsPerThread,
          bool         WithTile,
          unsigned int Trials = 100>
auto run_benchmark(benchmark::State& state, hipStream_t stream, size_t N)
    -> std::enable_if_t<std::is_same<Benchmark, subtract_left_partial>::value
                        || std::is_same<Benchmark, subtract_right_partial>::value>
{
    static constexpr auto items_per_block = BlockSize * ItemsPerThread;
    const auto num_blocks = (N + items_per_block - 1) / items_per_block;
    // Round up size to the next multiple of items_per_block
    const auto size = num_blocks * items_per_block;

    const std::vector<T> input = get_random_data<T>(size, T(0), T(10));
    const std::vector<unsigned int> tile_sizes
        = get_random_data<unsigned int>(num_blocks, 0, items_per_block);

    T*            d_input;
    unsigned int* d_tile_sizes;
    T*            d_output;
    HIP_CHECK(hipMalloc(&d_input, input.size() * sizeof(input[0])));
    HIP_CHECK(hipMalloc(&d_tile_sizes, tile_sizes.size() * sizeof(tile_sizes[0])));
    HIP_CHECK(hipMalloc(&d_output, input.size() * sizeof(input[0])));
    HIP_CHECK(
        hipMemcpy(
            d_input, input.data(),
            input.size() * sizeof(input[0]),
            hipMemcpyHostToDevice
        )
    );
    HIP_CHECK(
        hipMemcpy(
            d_tile_sizes, tile_sizes.data(),
            tile_sizes.size() * sizeof(tile_sizes[0]),
            hipMemcpyHostToDevice
        )
    );

    // HIP events creation
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    for(auto _ : state)
    {
        // Record start event
        HIP_CHECK(hipEventRecord(start, stream));

        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(kernel<Benchmark, BlockSize, ItemsPerThread, WithTile>),
            dim3(num_blocks), dim3(BlockSize), 0, stream,
            d_input, d_tile_sizes, d_output, Trials
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
    HIP_CHECK(hipFree(d_tile_sizes));
    HIP_CHECK(hipFree(d_output));
}

#define CREATE_BENCHMARK(T, BS, IPT, WITH_TILE)                                         \
    benchmark::RegisterBenchmark(                                                       \
        bench_naming::format_name("{lvl:block,algo:adjacent_difference,subalgo:" + name \
                                  + ",key_type:" #T ",cfg:{bs:" #BS ",ipt:" #IPT        \
                                    ",with_tile:" #WITH_TILE "}}")                      \
            .c_str(),                                                                   \
        run_benchmark<Benchmark, T, BS, IPT, WITH_TILE>,                                \
        stream,                                                                         \
        size)

#define BENCHMARK_TYPE(type, block, with_tile)    \
    CREATE_BENCHMARK(type, block, 1,  with_tile), \
    CREATE_BENCHMARK(type, block, 3,  with_tile), \
    CREATE_BENCHMARK(type, block, 4,  with_tile), \
    CREATE_BENCHMARK(type, block, 8,  with_tile), \
    CREATE_BENCHMARK(type, block, 16, with_tile), \
    CREATE_BENCHMARK(type, block, 32, with_tile)


template<class Benchmark>
void add_benchmarks(const std::string& name,
                    std::vector<benchmark::internal::Benchmark*>& benchmarks,
                    hipStream_t stream,
                    size_t size)
{
    std::vector<benchmark::internal::Benchmark*> bs =
    {
        BENCHMARK_TYPE(int, 256, false),
        BENCHMARK_TYPE(float, 256, false),
        BENCHMARK_TYPE(int8_t, 256, false),
        BENCHMARK_TYPE(rocprim::half, 256, false),
        BENCHMARK_TYPE(long long, 256, false),
        BENCHMARK_TYPE(double, 256, false)
    };

    if(!std::is_same<Benchmark, subtract_right_partial>::value) {
        bs.insert(bs.end(), {
            BENCHMARK_TYPE(int, 256, true),
            BENCHMARK_TYPE(float, 256, true),
            BENCHMARK_TYPE(int8_t, 256, true),
            BENCHMARK_TYPE(rocprim::half, 256, true),
            BENCHMARK_TYPE(long long, 256, true),
            BENCHMARK_TYPE(double, 256, true)
        });
    }

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
    add_benchmarks<subtract_left>("subtract_left", benchmarks, stream, size);
    add_benchmarks<subtract_right>("subtract_right", benchmarks, stream, size);
    add_benchmarks<subtract_left_partial>("subtract_left_partial", benchmarks, stream, size);
    add_benchmarks<subtract_right_partial>("subtract_right_partial", benchmarks, stream, size);

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
