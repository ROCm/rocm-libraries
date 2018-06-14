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
#include <rocprim/rocprim.hpp>

#include "benchmark_utils.hpp"

#ifndef DEFAULT_N
const size_t DEFAULT_N = 1024 * 1024 * 128;
#endif

const unsigned int batch_size = 10;
const unsigned int warmup_size = 5;

template<class T>
struct transform
{
    constexpr T operator()(const T& a) const [[hc]] [[cpu]]
    {
        return a + T(5);
    }
};

template<
    class T,
    class BinaryFunction
>
void run_benchmark(benchmark::State& state,
                   size_t size,
                   hc::accelerator_view acc_view,
                   BinaryFunction transform_op)
{
    std::vector<T> input = get_random_data<T>(size, T(0), T(1000));

    hc::array<T> d_input(hc::extent<1>(size), input.begin(), acc_view);
    hc::array<T> d_output(size, acc_view);
    acc_view.wait();

    // Warm-up
    for(size_t i = 0; i < warmup_size; i++)
    {
        rocprim::transform(
            d_input.accelerator_pointer(),
            d_output.accelerator_pointer(),
            size,
            transform_op,
            acc_view
        );
    }
    acc_view.wait();

    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();

        for(size_t i = 0; i < batch_size; i++)
        {
            rocprim::transform(
                d_input.accelerator_pointer(),
                d_output.accelerator_pointer(),
                size,
                transform_op,
                acc_view
            );
        }
        acc_view.wait();

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }
    state.SetBytesProcessed(state.iterations() * batch_size * size * sizeof(T));
    state.SetItemsProcessed(state.iterations() * batch_size * size);
}

#define CREATE_BENCHMARK(T, TRANSFORM_OP) \
benchmark::RegisterBenchmark( \
    ("transform<" #T ", " #TRANSFORM_OP ">"), \
    run_benchmark<T, TRANSFORM_OP>, size, acc_view, TRANSFORM_OP() \
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

    using custom_float2 = custom_type<float, float>;
    using custom_double2 = custom_type<double, double>;

    // Add benchmarks
    std::vector<benchmark::internal::Benchmark*> benchmarks =
    {
        CREATE_BENCHMARK(int, transform<int>),
        CREATE_BENCHMARK(long long, transform<long long>),

        CREATE_BENCHMARK(float, transform<float>),
        CREATE_BENCHMARK(double, transform<double>),

        CREATE_BENCHMARK(custom_float2, transform<custom_float2>),
        CREATE_BENCHMARK(custom_double2, transform<custom_double2>),
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
