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
#include <thread>
#include <vector>
#include <locale>
#include <codecvt>
#include <string>

// Google Benchmark
#include "benchmark/benchmark.h"

// HC API
#include <hcc/hc.hpp>

// rocPRIM HIP API
#include <rocprim/rocprim.hpp>

// CmdParser
#include "cmdparser.hpp"
#include "benchmark_utils.hpp"

#ifndef DEFAULT_N
const size_t DEFAULT_N = 1024 * 1024 * 128;
#endif

template<
    class T,
    class BinaryFunction
>
void run_benchmark(benchmark::State& state,
                   size_t size,
                   hc::accelerator_view acc_view,
                   BinaryFunction reduce_op)
{
    std::vector<T> input;
    std::vector<T> output(1);
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

    hc::array<T> d_input(hc::extent<1>(size), input.begin(), acc_view);
    hc::array<T> d_output(1, acc_view);
    acc_view.wait();

    // Allocate temporary storage memory
    size_t temp_storage_size_bytes;

    // Get size of d_temp_storage
    rocprim::reduce(
        nullptr,
        temp_storage_size_bytes,
        d_input.accelerator_pointer(),
        d_output.accelerator_pointer(),
        T(),
        size,
        reduce_op,
        acc_view
    );
    acc_view.wait();

    // allocate temporary storage
    hc::array<char> d_temp_storage(temp_storage_size_bytes, acc_view);
    acc_view.wait();

    // Warm-up
    for(size_t i = 0; i < 10; i++)
    {
        rocprim::reduce(
            d_temp_storage.accelerator_pointer(),
            temp_storage_size_bytes,
            d_input.accelerator_pointer(),
            d_output.accelerator_pointer(),
            T(),
            size,
            reduce_op,
            acc_view
        );
    }
    acc_view.wait();

    const unsigned int batch_size = 10;
    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();

        for(size_t i = 0; i < batch_size; i++)
        {
            rocprim::reduce(
                d_temp_storage.accelerator_pointer(),
                temp_storage_size_bytes,
                d_input.accelerator_pointer(),
                d_output.accelerator_pointer(),
                T(),
                size,
                reduce_op,
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

#define CREATE_BENCHMARK(T, REDUCE_OP) \
benchmark::RegisterBenchmark( \
    ("reduce<" #T "," #REDUCE_OP ">"), \
    run_benchmark<T, REDUCE_OP>, size, acc_view, REDUCE_OP() \
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

    // Add benchmarks
    std::vector<benchmark::internal::Benchmark*> benchmarks =
    {
        CREATE_BENCHMARK(unsigned int, rocprim::plus<unsigned int>),
        CREATE_BENCHMARK(unsigned long long, rocprim::plus<unsigned long long>),

        CREATE_BENCHMARK(float, rocprim::plus<float>),
        CREATE_BENCHMARK(double, rocprim::plus<double>),
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
