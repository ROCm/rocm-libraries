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

enum class benchmark_kinds
{
    sort_keys,
    sort_pairs
};

namespace rp = rocprim;

template<
    class T,
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    unsigned int Trials
>
void sort_keys(const hc::array<T>& input, hc::array<T>& output,
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

            T keys[ItemsPerThread];
            rp::block_load_direct_striped<BlockSize>(lid, input.data() + block_offset, keys);

            #pragma nounroll
            for(unsigned int trial = 0; trial < Trials; trial++)
            {
                rp::block_radix_sort<T, BlockSize, ItemsPerThread> sort;
                sort.sort(keys);
            }

            rp::block_store_direct_striped<BlockSize>(lid, output.data() + block_offset, keys);
        }
    );
}

template<
    class T,
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    unsigned int Trials
>
void sort_pairs(const hc::array<T>& input, hc::array<T>& output,
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

            T keys[ItemsPerThread];
            T values[ItemsPerThread];
            rp::block_load_direct_striped<BlockSize>(lid, input.data() + block_offset, keys);
            for(unsigned int i = 0; i < ItemsPerThread; i++)
            {
                values[i] = keys[i] + 1;
            }

            #pragma nounroll
            for(unsigned int trial = 0; trial < Trials; trial++)
            {
                rp::block_radix_sort<T, BlockSize, ItemsPerThread, T> sort;
                sort.sort(keys, values);
            }

            for(unsigned int i = 0; i < ItemsPerThread; i++)
            {
                keys[i] += values[i];
            }
            rp::block_store_direct_striped<BlockSize>(lid, output.data() + block_offset, keys);
        }
    );
}

template<
    class T,
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    unsigned int Trials = 10
>
void run_benchmark(
    benchmark::State& state,
    benchmark_kinds benchmark_kind,
    hc::accelerator_view acc_view, size_t N)
{
    constexpr auto items_per_block = BlockSize * ItemsPerThread;
    const auto size = items_per_block * ((N + items_per_block - 1)/items_per_block);

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

    hc::array<T> d_input(hc::extent<1>(input.size()), input.begin(), acc_view);
    hc::array<T> d_output(hc::extent<1>(input.size()),  acc_view);
    acc_view.wait();

    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();

        if(benchmark_kind == benchmark_kinds::sort_keys)
        {
            sort_keys<T, BlockSize, ItemsPerThread, Trials>(d_input, d_output, acc_view, size);
        }
        else if(benchmark_kind == benchmark_kinds::sort_pairs)
        {
            sort_pairs<T, BlockSize, ItemsPerThread, Trials>(d_input, d_output, acc_view, size);
        }
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
    (std::string("block_radix_sort<" #T ", " #BS ", " #IPT ">.") + name).c_str(), \
    run_benchmark<T, BS, IPT>, \
    benchmark_kind, acc_view, size \
)

void add_benchmarks(benchmark_kinds benchmark_kind,
                    const std::string& name,
                    std::vector<benchmark::internal::Benchmark*>& benchmarks,
                    hc::accelerator_view acc_view,
                    size_t size)
{
    std::vector<benchmark::internal::Benchmark*> bs =
    {
        CREATE_BENCHMARK(int, 64, 1),
        CREATE_BENCHMARK(int, 64, 2),
        CREATE_BENCHMARK(int, 64, 4),
        CREATE_BENCHMARK(int, 64, 7),
        CREATE_BENCHMARK(int, 64, 8),
        CREATE_BENCHMARK(int, 64, 11),
        CREATE_BENCHMARK(int, 64, 15),
        CREATE_BENCHMARK(int, 64, 19),
        CREATE_BENCHMARK(int, 64, 25),

        CREATE_BENCHMARK(int, 128, 1),
        CREATE_BENCHMARK(int, 128, 2),
        CREATE_BENCHMARK(int, 128, 4),
        CREATE_BENCHMARK(int, 128, 7),
        CREATE_BENCHMARK(int, 128, 8),
        CREATE_BENCHMARK(int, 128, 11),
        CREATE_BENCHMARK(int, 128, 15),
        CREATE_BENCHMARK(int, 128, 19),
        CREATE_BENCHMARK(int, 128, 25),

        CREATE_BENCHMARK(int, 256, 1),
        CREATE_BENCHMARK(int, 256, 2),
        CREATE_BENCHMARK(int, 256, 4),
        CREATE_BENCHMARK(int, 256, 7),
        CREATE_BENCHMARK(int, 256, 8),
        CREATE_BENCHMARK(int, 256, 11),
        CREATE_BENCHMARK(int, 256, 15),
        CREATE_BENCHMARK(int, 256, 19),
        CREATE_BENCHMARK(int, 256, 25),

        CREATE_BENCHMARK(int, 512, 1),
        CREATE_BENCHMARK(int, 512, 2),
        CREATE_BENCHMARK(int, 512, 4),
        CREATE_BENCHMARK(int, 512, 7),
        CREATE_BENCHMARK(int, 512, 8),

        CREATE_BENCHMARK(int, 1024, 1),
        CREATE_BENCHMARK(int, 1024, 2),
        CREATE_BENCHMARK(int, 1024, 4),

        CREATE_BENCHMARK(long long, 64, 1),
        CREATE_BENCHMARK(long long, 64, 2),
        CREATE_BENCHMARK(long long, 64, 4),
        CREATE_BENCHMARK(long long, 64, 7),
        CREATE_BENCHMARK(long long, 64, 8),
        CREATE_BENCHMARK(long long, 64, 11),
        CREATE_BENCHMARK(long long, 64, 15),
        CREATE_BENCHMARK(long long, 64, 19),
        CREATE_BENCHMARK(long long, 64, 25),

        CREATE_BENCHMARK(long long, 128, 1),
        CREATE_BENCHMARK(long long, 128, 2),
        CREATE_BENCHMARK(long long, 128, 4),
        CREATE_BENCHMARK(long long, 128, 7),
        CREATE_BENCHMARK(long long, 128, 8),
        CREATE_BENCHMARK(long long, 128, 11),
        CREATE_BENCHMARK(long long, 128, 15),
        CREATE_BENCHMARK(long long, 128, 19),
        CREATE_BENCHMARK(long long, 128, 25),

        CREATE_BENCHMARK(long long, 256, 1),
        CREATE_BENCHMARK(long long, 256, 2),
        CREATE_BENCHMARK(long long, 256, 4),
        CREATE_BENCHMARK(long long, 256, 7),
        CREATE_BENCHMARK(long long, 256, 8),
        CREATE_BENCHMARK(long long, 256, 11),
        CREATE_BENCHMARK(long long, 256, 15),
        CREATE_BENCHMARK(long long, 256, 19),

        CREATE_BENCHMARK(long long, 512, 1),
        CREATE_BENCHMARK(long long, 512, 2),
        CREATE_BENCHMARK(long long, 512, 4),
        CREATE_BENCHMARK(long long, 512, 7),
        CREATE_BENCHMARK(long long, 512, 8),

        CREATE_BENCHMARK(long long, 1024, 1),
        CREATE_BENCHMARK(long long, 1024, 2),
        CREATE_BENCHMARK(long long, 1024, 4),
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

    // HC
    hc::accelerator acc;
    auto acc_view = acc.get_default_view();

    // Add benchmarks
    std::vector<benchmark::internal::Benchmark*> benchmarks;
    add_benchmarks(benchmark_kinds::sort_keys, "sort(keys)", benchmarks, acc_view, size);
    add_benchmarks(benchmark_kinds::sort_pairs, "sort(keys, values)", benchmarks, acc_view, size);

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
