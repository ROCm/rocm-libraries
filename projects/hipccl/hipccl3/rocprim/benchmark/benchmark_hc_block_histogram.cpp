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
#include <rocprim/block/block_histogram.hpp>

#ifndef DEFAULT_N
const size_t DEFAULT_N = 1024 * 1024 * 128;
#endif

namespace rp = rocprim;

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
    static void run(const hc::array<T>& input, hc::array<T>& output,
                    hc::accelerator_view acc_view, size_t size)
    {
        const size_t grid_size = size / ItemsPerThread;
        hc::parallel_for_each(
            acc_view,
            hc::extent<1>(grid_size).tile(BlockSize),
            [&](hc::tiled_index<1> idx) [[hc]]
            {
                const unsigned int index = ((idx.tile[0] * BlockSize) + idx.local[0]) * ItemsPerThread;
                unsigned int global_offset = idx.tile[0] * BinSize;

                T values[ItemsPerThread];
                for(unsigned int k = 0; k < ItemsPerThread; k++)
                {
                    values[k] = input[index + k];
                }

                using bhistogram_t =  rp::block_histogram<T, BlockSize, ItemsPerThread, BinSize, algorithm>;
                tile_static T histogram[BinSize];
                tile_static typename bhistogram_t::storage_type storage;

                #pragma nounroll
                for(unsigned int trial = 0; trial < Trials; trial++)
                {
                    bhistogram_t().histogram(values, histogram, storage);
                }

                #pragma unroll
                for (unsigned int offset = 0; offset < BinSize; offset += BlockSize)
                {
                    if(offset + idx.local[0] < BinSize)
                    {
                        output[global_offset + idx.local[0]] = histogram[offset + idx.local[0]];
                        global_offset += BlockSize;
                    }
                }
            }
        );
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
void run_benchmark(benchmark::State& state, hc::accelerator_view acc_view, size_t N)
{
    // Make sure size is a multiple of BlockSize
    constexpr auto items_per_block = BlockSize * ItemsPerThread;
    const auto size = items_per_block * ((N + items_per_block - 1)/items_per_block);
    const auto bin_size = BinSize * ((N + items_per_block - 1)/items_per_block);
    // Allocate and fill memory
    std::vector<T> input(size, 0.0f);

    hc::array<T> d_input(hc::extent<1>(input.size()), input.begin(), acc_view);
    hc::array<T> d_output(hc::extent<1>(bin_size * sizeof(T)), acc_view);
    acc_view.wait();

    for (auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();

        Benchmark::template run<T, BlockSize, ItemsPerThread, BinSize, Trials>
            (d_input, d_output, acc_view, size);
        acc_view.wait();

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

        state.SetIterationTime(elapsed_seconds.count());
    }
    state.SetBytesProcessed(state.iterations() * size * sizeof(T) * Trials);
    state.SetItemsProcessed(state.iterations() * size * Trials);
}

// IPT - items per thread
#define CREATE_BENCHMARK(T, BS, IPT) \
    benchmark::RegisterBenchmark( \
        (std::string("block_histogram<"#T", "#BS", "#IPT", " + algorithm_name + ">.") + method_name).c_str(), \
        run_benchmark<Benchmark, T, BS, IPT>, \
        acc_view, size \
    )

template<class Benchmark>
void add_benchmarks(std::vector<benchmark::internal::Benchmark*>& benchmarks,
                    const std::string& method_name,
                    const std::string& algorithm_name,
                    hc::accelerator_view acc_view,
                    size_t size)
{
    std::vector<benchmark::internal::Benchmark*> new_benchmarks =
    {
        CREATE_BENCHMARK(int, 256, 1),
        CREATE_BENCHMARK(int, 256, 2),
        CREATE_BENCHMARK(int, 256, 3),
        CREATE_BENCHMARK(int, 256, 4),
        CREATE_BENCHMARK(int, 256, 8),
        CREATE_BENCHMARK(int, 256, 11),
        CREATE_BENCHMARK(int, 256, 16),

        CREATE_BENCHMARK(int, 320, 1),
        CREATE_BENCHMARK(int, 320, 2),
        CREATE_BENCHMARK(int, 320, 3),
        CREATE_BENCHMARK(int, 320, 4),
        CREATE_BENCHMARK(int, 320, 8),
        CREATE_BENCHMARK(int, 320, 11),
        CREATE_BENCHMARK(int, 320, 16)
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

    // HC
    hc::accelerator acc;
    auto acc_view = acc.get_default_view();

    // Add benchmarks
    std::vector<benchmark::internal::Benchmark*> benchmarks;
    // using_atomic
    using histogram_a_t = histogram<rocprim::block_histogram_algorithm::using_atomic>;
    add_benchmarks<histogram_a_t>(
        benchmarks, "histogram", "using_atomic", acc_view, size
    );
    // using_sort
    using histogram_s_t = histogram<rocprim::block_histogram_algorithm::using_sort>;
    add_benchmarks<histogram_s_t>(
        benchmarks, "histogram", "using_sort", acc_view, size
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
