// MIT License
//
// Copyright (c) 2017-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <cstddef>
#include <iostream>
#include <string>

// Google Benchmark
#include <benchmark/benchmark.h>

// HIP API
#include <hip/hip_runtime.h>

// CmdParser
#include "cmdparser.hpp"

#include "benchmark_block_sort.parallel.hpp"
#include "benchmark_utils.hpp"

#ifndef DEFAULT_N
const size_t DEFAULT_N = 1024 * 1024 * 128;
#endif

#define CREATE_BENCHMARK_IPT(K, V, BS, IPT)                                                        \
    config_autotune_register::create<                                                              \
        block_sort_benchmark<K, V, BS, IPT, rocprim::block_sort_algorithm::merge_sort, true>>();   \
    config_autotune_register::create<                                                              \
        block_sort_benchmark<K, V, BS, IPT, rocprim::block_sort_algorithm::merge_sort, false>>();  \
    config_autotune_register::create<                                                              \
        block_sort_benchmark<K,                                                                    \
                             V,                                                                    \
                             BS,                                                                   \
                             IPT,                                                                  \
                             rocprim::block_sort_algorithm::stable_merge_sort,                     \
                             true>>();                                                             \
    config_autotune_register::create<                                                              \
        block_sort_benchmark<K,                                                                    \
                             V,                                                                    \
                             BS,                                                                   \
                             IPT,                                                                  \
                             rocprim::block_sort_algorithm::stable_merge_sort,                     \
                             false>>();                                                            \
    config_autotune_register::create<                                                              \
        block_sort_benchmark<K, V, BS, IPT, rocprim::block_sort_algorithm::bitonic_sort, true>>(); \
    config_autotune_register::create<                                                              \
        block_sort_benchmark<K,                                                                    \
                             V,                                                                    \
                             BS,                                                                   \
                             IPT,                                                                  \
                             rocprim::block_sort_algorithm::bitonic_sort,                          \
                             false>>();

#define CREATE_BENCHMARK(K, V, BS)    \
    CREATE_BENCHMARK_IPT(K, V, BS, 1) \
    CREATE_BENCHMARK_IPT(K, V, BS, 4)

int main(int argc, char* argv[])
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
    const size_t size   = parser.get<size_t>("size");
    const int    trials = parser.get<int>("trials");
    bench_naming::set_format(parser.get<std::string>("name_format"));

    // HIP
    const hipStream_t stream = 0; // default

    // Benchmark info
    add_common_benchmark_info();
    benchmark::AddCustomContext("size", std::to_string(size));

// If we are NOT config tuning run a selection of benchmarks
// Block sizes as large as possible ar most relevant
#ifndef BENCHMARK_CONFIG_TUNING
    CREATE_BENCHMARK(float, rocprim::empty_type, 256)
    CREATE_BENCHMARK(double, rocprim::empty_type, 256)
    CREATE_BENCHMARK(rocprim::half, rocprim::empty_type, 256)
    CREATE_BENCHMARK(uint8_t, rocprim::empty_type, 256)
    CREATE_BENCHMARK(int, rocprim::empty_type, 256)
    CREATE_BENCHMARK(int, rocprim::empty_type, 512)
    CREATE_BENCHMARK(double, rocprim::empty_type, 512)
    CREATE_BENCHMARK(int, int, 512)
    CREATE_BENCHMARK(float, double, 512)
    CREATE_BENCHMARK(double, int64_t, 512)
    CREATE_BENCHMARK(rocprim::half, int16_t, 512)
    CREATE_BENCHMARK(uint8_t, uint32_t, 512)
#endif

    std::vector<benchmark::internal::Benchmark*> benchmarks = {};
    config_autotune_register::register_benchmark_subset(benchmarks, 0, 1, size, stream);

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
