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

// rocPRIM
#include <rocprim/rocprim.hpp>

#ifndef DEFAULT_N
const size_t DEFAULT_N = 1024 * 1024 * 128;
#endif

namespace rp = rocprim;

struct blocked_to_striped
{
    template<
        class T,
        unsigned int BlockSize,
        unsigned int ItemsPerThread,
        unsigned int Trials
    >
    static void run(hc::array<T> & d_input, hc::array<T> & d_output,
                    hc::accelerator_view acc_view, size_t size)
    {
        const size_t grid_size = size / ItemsPerThread;
        hc::parallel_for_each(
            acc_view,
            hc::extent<1>(grid_size).tile(BlockSize),
            [&](hc::tiled_index<1> idx) [[hc]]
            {
                const unsigned int lid = idx.local[0];
                const unsigned int block_offset = idx.tile[0] * ItemsPerThread * BlockSize;

                T input[ItemsPerThread];
                rp::block_load_direct_striped<BlockSize>(lid, d_input.data() + block_offset, input);

                #pragma nounroll
                for(unsigned int trial = 0; trial < Trials; trial++)
                {
                    rp::block_exchange<T, BlockSize, ItemsPerThread> exchange;
                    exchange.blocked_to_striped(input, input);
                }

                rp::block_store_direct_striped<BlockSize>(lid, d_output.data() + block_offset, input);
            }
        );
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
    static void run(hc::array<T> & d_input, hc::array<T> & d_output,
                    hc::accelerator_view acc_view, size_t size)
    {
        const size_t grid_size = size / ItemsPerThread;
        hc::parallel_for_each(
            acc_view,
            hc::extent<1>(grid_size).tile(BlockSize),
            [&](hc::tiled_index<1> idx) [[hc]]
            {
                const unsigned int lid = idx.local[0];
                const unsigned int block_offset = idx.tile[0] * ItemsPerThread * BlockSize;

                T input[ItemsPerThread];
                rp::block_load_direct_striped<BlockSize>(lid, d_input.data() + block_offset, input);

                #pragma nounroll
                for(unsigned int trial = 0; trial < Trials; trial++)
                {
                    rp::block_exchange<T, BlockSize, ItemsPerThread> exchange;
                    exchange.striped_to_blocked(input, input);
                }

                rp::block_store_direct_striped<BlockSize>(lid, d_output.data() + block_offset, input);
            }
        );
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
    static void run(hc::array<T> & d_input, hc::array<T> & d_output,
                    hc::accelerator_view acc_view, size_t size)
    {
        const size_t grid_size = size / ItemsPerThread;
        hc::parallel_for_each(
            acc_view,
            hc::extent<1>(grid_size).tile(BlockSize),
            [&](hc::tiled_index<1> idx) [[hc]]
            {
                const unsigned int lid = idx.local[0];
                const unsigned int block_offset = idx.tile[0] * ItemsPerThread * BlockSize;

                T input[ItemsPerThread];
                rp::block_load_direct_striped<BlockSize>(lid, d_input.data() + block_offset, input);

                #pragma nounroll
                for(unsigned int trial = 0; trial < Trials; trial++)
                {
                    rp::block_exchange<T, BlockSize, ItemsPerThread> exchange;
                    exchange.blocked_to_warp_striped(input, input);
                }

                rp::block_store_direct_striped<BlockSize>(lid, d_output.data() + block_offset, input);
            }
        );
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
    static void run(hc::array<T> & d_input, hc::array<T> & d_output,
                    hc::accelerator_view acc_view, size_t size)
    {
        const size_t grid_size = size / ItemsPerThread;
        hc::parallel_for_each(
            acc_view,
            hc::extent<1>(grid_size).tile(BlockSize),
            [&](hc::tiled_index<1> idx) [[hc]]
            {
                const unsigned int lid = idx.local[0];
                const unsigned int block_offset = idx.tile[0] * ItemsPerThread * BlockSize;

                T input[ItemsPerThread];
                rp::block_load_direct_striped<BlockSize>(lid, d_input.data() + block_offset, input);

                #pragma nounroll
                for(unsigned int trial = 0; trial < Trials; trial++)
                {
                    rp::block_exchange<T, BlockSize, ItemsPerThread> exchange;
                    exchange.warp_striped_to_blocked(input, input);
                }

                rp::block_store_direct_striped<BlockSize>(lid, d_output.data() + block_offset, input);
            }
        );
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
    static void run(hc::array<T> & d_input, hc::array<T> & d_output,
                    hc::accelerator_view acc_view, size_t size)
    {
        const size_t grid_size = size / ItemsPerThread;
        hc::parallel_for_each(
            acc_view,
            hc::extent<1>(grid_size).tile(BlockSize),
            [&](hc::tiled_index<1> idx) [[hc]]
            {
                const unsigned int lid = idx.local[0];
                const unsigned int block_offset = idx.tile[0] * ItemsPerThread * BlockSize;
                T input[ItemsPerThread];
                unsigned int ranks[ItemsPerThread];
                rp::block_load_direct_striped<BlockSize>(lid, d_input.data() + block_offset, input);
                rp::block_load_direct_striped<BlockSize>(lid, d_input.data() + block_offset, ranks);

                #pragma nounroll
                for(unsigned int trial = 0; trial < Trials; trial++)
                {
                    rp::block_exchange<T, BlockSize, ItemsPerThread> exchange;
                    exchange.scatter_to_blocked(input, input, ranks);
                }

                rp::block_store_direct_striped<BlockSize>(lid, d_output.data() + block_offset, input);
            }
        );
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
    static void run(hc::array<T> & d_input, hc::array<T> & d_output,
                    hc::accelerator_view acc_view, size_t size)
    {
        const size_t grid_size = size / ItemsPerThread;
        hc::parallel_for_each(
            acc_view,
            hc::extent<1>(grid_size).tile(BlockSize),
            [&](hc::tiled_index<1> idx) [[hc]]
            {
                const unsigned int lid = idx.local[0];
                const unsigned int block_offset = idx.tile[0] * ItemsPerThread * BlockSize;
                T input[ItemsPerThread];
                unsigned int ranks[ItemsPerThread];
                rp::block_load_direct_striped<BlockSize>(lid, d_input.data() + block_offset, input);
                rp::block_load_direct_striped<BlockSize>(lid, d_input.data() + block_offset, ranks);

                #pragma nounroll
                for(unsigned int trial = 0; trial < Trials; trial++)
                {
                    rp::block_exchange<T, BlockSize, ItemsPerThread> exchange;
                    exchange.scatter_to_striped(input, input, ranks);
                }

                rp::block_store_direct_striped<BlockSize>(lid, d_output.data() + block_offset, input);
            }
        );
    }
};

template<
    class Benchmark,
    class T,
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    unsigned int Trials = 100
>
void run_benchmark(benchmark::State& state, hc::accelerator_view acc_view, size_t N)
{
    constexpr auto items_per_block = BlockSize * ItemsPerThread;
    const auto size = items_per_block * ((N + items_per_block - 1)/items_per_block);

    std::vector<T> input(size);
    std::vector<T> output(size);
    // Fill input with ranks (for scatter operations)
    std::mt19937 gen;
    for(size_t bi = 0; bi < size / items_per_block; bi++)
    {
        auto block_ranks = input.begin() + bi * items_per_block;
        std::iota(block_ranks, block_ranks + items_per_block, 0);
        std::shuffle(block_ranks, block_ranks + items_per_block, gen);
    }

    hc::array<T> d_input(hc::extent<1>(input.size()), input.begin(), acc_view);
    hc::array<T> d_output(hc::extent<1>(output.size()), output.begin(), acc_view);
    acc_view.wait();

    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();

        Benchmark::template run<T, BlockSize, ItemsPerThread, Trials>
            (d_input, d_output, acc_view, size);

        acc_view.wait();

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }
    state.SetBytesProcessed(state.iterations() * Trials * size * sizeof(T));
    state.SetItemsProcessed(state.iterations() * Trials * size);
}

#define CREATE_BENCHMARK(T, BS, IPT) \
benchmark::RegisterBenchmark( \
    (std::string("block_exchange<" #T ", " #BS ", " #IPT ">.") + name).c_str(), \
    run_benchmark<Benchmark, T, BS, IPT>, \
    acc_view, size \
)

template<class Benchmark>
void add_benchmarks(const std::string& name,
                    std::vector<benchmark::internal::Benchmark*>& benchmarks,
                    hc::accelerator_view acc_view,
                    size_t size)
{
    std::vector<benchmark::internal::Benchmark*> bs =
    {
        CREATE_BENCHMARK(int, 256, 1),
        CREATE_BENCHMARK(int, 256, 2),
        CREATE_BENCHMARK(int, 256, 3),
        CREATE_BENCHMARK(int, 256, 4),
        CREATE_BENCHMARK(int, 256, 6),
        CREATE_BENCHMARK(int, 256, 7),
        CREATE_BENCHMARK(int, 256, 8),
        CREATE_BENCHMARK(long long, 256, 1),
        CREATE_BENCHMARK(long long, 256, 2),
        CREATE_BENCHMARK(long long, 256, 3),
        CREATE_BENCHMARK(long long, 256, 4),
        CREATE_BENCHMARK(long long, 256, 6),
        CREATE_BENCHMARK(long long, 256, 7),
        CREATE_BENCHMARK(long long, 256, 8),
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
    hc::accelerator acc;
    auto acc_view = acc.get_default_view();

    // Add benchmarks
    std::vector<benchmark::internal::Benchmark*> benchmarks;
    add_benchmarks<blocked_to_striped>("blocked_to_striped", benchmarks, acc_view, size);
    add_benchmarks<striped_to_blocked>("striped_to_blocked", benchmarks, acc_view, size);
    add_benchmarks<blocked_to_warp_striped>("blocked_to_warp_striped", benchmarks, acc_view, size);
    add_benchmarks<warp_striped_to_blocked>("warp_striped_to_blocked", benchmarks, acc_view, size);
    add_benchmarks<scatter_to_blocked>("scatter_to_blocked", benchmarks, acc_view, size);
    add_benchmarks<scatter_to_striped>("scatter_to_striped", benchmarks, acc_view, size);

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
