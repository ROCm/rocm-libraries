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
#include <locale>
#include <codecvt>
#include <string>
#include <cstdio>
#include <cstdlib>

// Google Benchmark
#include "benchmark/benchmark.h"
// CmdParser
#include "cmdparser.hpp"
// HC API
#include <hcc/hc.hpp>
// rocPRIM
#include <rocprim/warp/warp_sort.hpp>

#include "benchmark_utils.hpp"

#define HIP_CHECK(condition)         \
  {                                  \
    hipError_t error = condition;    \
    if(error != hipSuccess){         \
        std::cout << "HIP error: " << error << " line: " << __LINE__ << std::endl; \
        exit(error); \
    } \
  }

#ifndef DEFAULT_N
const size_t DEFAULT_N = 1024 * 1024 * 32;
#endif

namespace rp = rocprim;

template<
    class Key,
    unsigned int BlockSize,
    unsigned int WarpSize,
    class Value = Key,
    bool SortByKey = false,
    unsigned int Trials = 100
>
void run_benchmark(benchmark::State& state, hc::accelerator_view acc_view, size_t size)
{
    // Make sure size is a multiple of BlockSize
    size = BlockSize * ((size + BlockSize - 1)/BlockSize);

    // Create data on host
    std::vector<Key> input_key = get_random_data(size, Key(0), Key(10000));
    std::vector<Value> input_value(size_t(1));
    if(SortByKey) input_value = get_random_data(size, Value(0), Value(10000));

    // Transfer to device
    hc::array_view<Key, 1> av_input_key(input_key.size(), input_key.data());
    hc::array_view<Value, 1> av_input_value(input_value.size(), input_value.data());
    av_input_key.synchronize_to(acc_view);
    av_input_value.synchronize_to(acc_view);
    acc_view.wait();

    for (auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();
        auto event = hc::parallel_for_each(
            acc_view,
            hc::extent<1>(size).tile(BlockSize),
            [=](hc::tiled_index<1> i) [[hc]]
            {
                auto key = av_input_key[i];
                if(SortByKey)
                {
                    auto value = av_input_value[i];
                    rp::warp_sort<Key, WarpSize, Value> wsort;
                    #pragma nounroll
                    for(unsigned int trial = 0; trial < Trials; trial++)
                    {
                        wsort.sort(key, value);
                    }
                    av_input_value[i] = value;
                }
                else
                {
                    rp::warp_sort<Key, WarpSize> wsort;
                    #pragma nounroll
                    for(unsigned int trial = 0; trial < Trials; trial++)
                    {
                        wsort.sort(key);
                    }
                }
                av_input_key[i] = key;
            }
        );
        event.wait();

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }
    // SortByKey also transfers values
    auto sorted_type_size = sizeof(Key);
    if(SortByKey) sorted_type_size += sizeof(Value);
    state.SetBytesProcessed(state.iterations() * size * sorted_type_size * Trials);
    state.SetItemsProcessed(state.iterations() * size * Trials);
}

#define CREATE_SORT_BENCHMARK(K, BS, WS) \
    benchmark::RegisterBenchmark( \
        "warp_sort<"#K", "#BS", "#WS">.sort(only keys)", \
        run_benchmark<K, BS, WS>, \
        acc_view, size \
    )

#define CREATE_SORTBYKEY_BENCHMARK(K, V, BS, WS) \
    benchmark::RegisterBenchmark( \
        "warp_sort<"#K", "#BS", "#WS", "#V">.sort", \
        run_benchmark<K, BS, WS, V, true>, \
        acc_view, size \
    )

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

    using custom_double2 = custom_type<double, double>;
    using custom_int_double = custom_type<int, double>;
    std::vector<benchmark::internal::Benchmark*> benchmarks =
    {
        // key type, block size, warp size
        CREATE_SORT_BENCHMARK(float, 64, 64),
        CREATE_SORT_BENCHMARK(float, 128, 64),
        CREATE_SORT_BENCHMARK(float, 256, 64),
        CREATE_SORT_BENCHMARK(float, 64, 32),
        CREATE_SORT_BENCHMARK(float, 64, 16),

        CREATE_SORT_BENCHMARK(int, 64, 64),
        CREATE_SORT_BENCHMARK(double, 64, 64),
        CREATE_SORT_BENCHMARK(custom_double2, 64, 64),
        CREATE_SORT_BENCHMARK(custom_int_double , 64, 64),

        // key type, value type, block size, warp size
        CREATE_SORTBYKEY_BENCHMARK(float, float, 64, 64),
        CREATE_SORTBYKEY_BENCHMARK(float, float, 256, 64),

        CREATE_SORTBYKEY_BENCHMARK(unsigned int, int, 64, 64),
        CREATE_SORTBYKEY_BENCHMARK(int, double, 64, 64),
        CREATE_SORTBYKEY_BENCHMARK(int, custom_double2, 64, 64),
        CREATE_SORTBYKEY_BENCHMARK(int, custom_int_double, 64, 64),
    };

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
