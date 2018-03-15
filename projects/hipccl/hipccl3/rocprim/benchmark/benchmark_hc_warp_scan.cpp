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

// Google Benchmark
#include "benchmark/benchmark.h"
// CmdParser
#include "cmdparser.hpp"
// HC API
#include <hcc/hc.hpp>
// rocPRIM
#include <rocprim/warp/warp_scan.hpp>

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
    class T,
    unsigned int BlockSize,
    unsigned int WarpSize,
    bool Inclusive = true,
    unsigned int Trials = 100
>
void run_benchmark(benchmark::State& state, hc::accelerator_view acc_view, size_t size)
{
    // Make sure size is a multiple of BlockSize
    size = BlockSize * ((size + BlockSize - 1)/BlockSize);
    // Allocate and fill memory
    std::vector<T> input(size, T(1));
    std::vector<T> output(size);
    hc::array_view<T, 1> av_input(size, input.data());
    hc::array_view<T, 1> av_output(size, output.data());
    av_input.synchronize_to(acc_view);
    av_output.synchronize_to(acc_view);
    acc_view.wait();

    const auto init = input[0];
    for (auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();
        auto event = hc::parallel_for_each(
            acc_view,
            hc::extent<1>(size).tile(BlockSize),
            [=](hc::tiled_index<1> i) [[hc]]
            {
                T value = av_input[i];

                using wscan_t = rp::warp_scan<T, WarpSize>;
                tile_static typename wscan_t::storage_type storage;
                if(Inclusive)
                {
                    #pragma nounroll
                    for(unsigned int trial = 0; trial < Trials; trial++)
                    {
                        wscan_t().inclusive_scan(value, value, storage);
                    }
                }
                else
                {
                    #pragma nounroll
                    for(unsigned int trial = 0; trial < Trials; trial++)
                    {
                        wscan_t().exclusive_scan(value, value, init, storage);
                    }
                }

                av_output[i] = value;
            }
        );
        event.wait();
        acc_view.wait();

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

        state.SetIterationTime(elapsed_seconds.count());
    }
    state.SetBytesProcessed(state.iterations() * size * sizeof(T) * Trials);
    state.SetItemsProcessed(state.iterations() * size * Trials);
}

#define CREATE_BENCHMARK(T, BS, WS, INCLUSIVE) \
    benchmark::RegisterBenchmark( \
        (std::string("warp_scan<"#T", "#BS", "#WS">.") + method_name).c_str(), \
        run_benchmark<T, BS, WS, INCLUSIVE>, \
        acc_view, size \
    )

template<bool Inclusive>
void add_benchmarks(std::vector<benchmark::internal::Benchmark*>& benchmarks,
                    const std::string& method_name,
                    hc::accelerator_view acc_view,
                    size_t size)
{
    using custom_double2 = custom_type<double, double>;
    using custom_int_double = custom_type<int, double>;

    std::vector<benchmark::internal::Benchmark*> new_benchmarks =
    {
        CREATE_BENCHMARK(float, 64, 64, Inclusive),
        CREATE_BENCHMARK(float, 128, 64, Inclusive),
        CREATE_BENCHMARK(float, 256, 64, Inclusive),
        CREATE_BENCHMARK(float, 256, 32, Inclusive),
        CREATE_BENCHMARK(float, 256, 16, Inclusive),
        // force using shared memory version
        CREATE_BENCHMARK(float, 63, 63, Inclusive),
        CREATE_BENCHMARK(float, 62, 31, Inclusive),
        CREATE_BENCHMARK(float, 60, 15, Inclusive),

        CREATE_BENCHMARK(int, 64, 64, Inclusive),
        CREATE_BENCHMARK(int, 128, 64, Inclusive),
        CREATE_BENCHMARK(int, 256, 64, Inclusive),

        CREATE_BENCHMARK(double, 64, 64, Inclusive),
        CREATE_BENCHMARK(double, 128, 64, Inclusive),
        CREATE_BENCHMARK(double, 256, 64, Inclusive),

        CREATE_BENCHMARK(custom_double2, 64, 64, Inclusive),
        CREATE_BENCHMARK(custom_int_double, 64, 64, Inclusive)
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
    std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;
    std::cout << "[HC]  Device name: " << conv.to_bytes(acc.get_description()) << std::endl;

    // Add benchmarks
    std::vector<benchmark::internal::Benchmark*> benchmarks;
    add_benchmarks<true>(benchmarks, "inclusive_scan", acc_view, size);
    add_benchmarks<false>(benchmarks, "exclusive_scan", acc_view, size);

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
