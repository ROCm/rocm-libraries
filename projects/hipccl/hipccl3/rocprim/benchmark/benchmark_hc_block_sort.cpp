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
#include <codecvt>
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
#include <hip/hip_hcc.h>

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

enum class benchmark_kinds
{
    sort_keys,
    sort_pairs
};

namespace rp = rocprim;

template<
    class T,
    unsigned int BlockSize,
    unsigned int Trials = 10
>
void run_benchmark(benchmark::State& state, benchmark_kinds benchmark_kind, hc::accelerator_view acc_view, size_t N)
{
    constexpr auto block_size = BlockSize;
    const auto size = block_size * ((N + block_size - 1)/block_size);

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
    hc::array_view<T, 1> d_input(input.size(), input.data());
    hc::array_view<T, 1> d_output(input.size());
    d_input.synchronize_to(acc_view);
    acc_view.wait();

    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();

        if(benchmark_kind == benchmark_kinds::sort_keys)
        {
            auto event = hc::parallel_for_each(
                        hc::extent<1>(size).tile(block_size),
                        [=](hc::tiled_index<1> i) [[hc]]
                        {
                            T key = d_input[i];

                            #pragma nounroll
                            for(unsigned int trial = 0; trial < Trials; trial++)
                            {
                                rp::block_sort<T, BlockSize> bsort;
                                bsort.sort(key);
                            }

                            d_output[i] = key;
                        }
                    );
            event.wait();
        }
        else if(benchmark_kind == benchmark_kinds::sort_pairs)
        {
            auto event = hc::parallel_for_each(
                        hc::extent<1>(size).tile(block_size),
                        [=](hc::tiled_index<1> i) [[hc]]
                        {
                            T key = d_input[i];
                            T value = key + 1;

                            #pragma nounroll
                            for(unsigned int trial = 0; trial < Trials; trial++)
                            {
                                rp::block_sort<T, BlockSize, T> bsort;
                                bsort.sort(key, value);
                            }

                            d_output[i] = key + value;
                        }
                    );
            event.wait();
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }
    state.SetBytesProcessed(state.iterations() * Trials * size * sizeof(T));
    state.SetItemsProcessed(state.iterations() * Trials * size);
}

#define CREATE_BENCHMARK(T, BS) \
benchmark::RegisterBenchmark( \
    (std::string("block_sort<" #T ", " #BS ">.") + name).c_str(), \
    run_benchmark<T, BS>, \
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
        CREATE_BENCHMARK(int, 64),
        CREATE_BENCHMARK(int, 128),
        CREATE_BENCHMARK(int, 192),
        CREATE_BENCHMARK(int, 256),
        CREATE_BENCHMARK(int, 320),
        CREATE_BENCHMARK(int, 512),
        CREATE_BENCHMARK(int, 1024),

        CREATE_BENCHMARK(long long, 64),
        CREATE_BENCHMARK(long long, 128),
        CREATE_BENCHMARK(long long, 192),
        CREATE_BENCHMARK(long long, 256),
        CREATE_BENCHMARK(long long, 320),
        CREATE_BENCHMARK(long long, 512),
        CREATE_BENCHMARK(long long, 1024)
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
    std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;
    std::cout << "[HC]  Device name: " << conv.to_bytes(acc.get_description()) << std::endl;

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
