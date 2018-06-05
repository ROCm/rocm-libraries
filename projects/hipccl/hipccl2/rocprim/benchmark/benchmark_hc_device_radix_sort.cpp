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
#include <locale>
#include <codecvt>

// Google Benchmark
#include "benchmark/benchmark.h"
// CmdParser
#include "cmdparser.hpp"
#include "benchmark_utils.hpp"

// HC API
#include <hcc/hc.hpp>

// rocPRIM
#include <rocprim/rocprim.hpp>

#ifndef DEFAULT_N
const size_t DEFAULT_N = 1024 * 1024 * 32;
#endif

enum class benchmark_kinds
{
    sort_keys,
    sort_keys_desc,
    sort_pairs,
    sort_pairs_desc
};

namespace rp = rocprim;

template<benchmark_kinds benchmark_kind, class Key, class Value>
auto run_radix_sort(void * d_temporary_storage, size_t& temporary_storage_bytes,
                           Key * d_keys_input, Key * d_keys_output,
                           Value *, Value *,
                           size_t size,
                           hc::accelerator_view acc_view)
    -> typename std::enable_if<benchmark_kind == benchmark_kinds::sort_keys, void>::type
{
    return rp::radix_sort_keys(
        d_temporary_storage, temporary_storage_bytes,
        d_keys_input, d_keys_output, size,
        0, sizeof(Key) * 8,
        acc_view
    );
}

template<benchmark_kinds benchmark_kind, class Key, class Value>
auto run_radix_sort(void * d_temporary_storage, size_t& temporary_storage_bytes,
                           Key * d_keys_input, Key * d_keys_output,
                           Value *, Value *,
                           size_t size,
                           hc::accelerator_view acc_view)
    -> typename std::enable_if<benchmark_kind == benchmark_kinds::sort_keys_desc, void>::type
{
    return rp::radix_sort_keys_desc(
        d_temporary_storage, temporary_storage_bytes,
        d_keys_input, d_keys_output, size,
        0, sizeof(Key) * 8,
        acc_view
    );
}

template<benchmark_kinds benchmark_kind, class Key, class Value>
auto run_radix_sort(void * d_temporary_storage, size_t& temporary_storage_bytes,
                           Key * d_keys_input, Key * d_keys_output,
                           Value * d_values_input, Value * d_values_output,
                           size_t size,
                           hc::accelerator_view acc_view)
    -> typename std::enable_if<benchmark_kind == benchmark_kinds::sort_pairs, void>::type
{
    return rp::radix_sort_pairs(
        d_temporary_storage, temporary_storage_bytes,
        d_keys_input, d_keys_output, d_values_input, d_values_output, size,
        0, sizeof(Key) * 8,
        acc_view
    );
}

template<benchmark_kinds benchmark_kind, class Key, class Value>
auto run_radix_sort(void * d_temporary_storage, size_t& temporary_storage_bytes,
                           Key * d_keys_input, Key * d_keys_output,
                           Value * d_values_input, Value * d_values_output,
                           size_t size,
                           hc::accelerator_view acc_view)
    -> typename std::enable_if<benchmark_kind == benchmark_kinds::sort_pairs_desc, void>::type
{
    return rp::radix_sort_pairs_desc(
        d_temporary_storage, temporary_storage_bytes,
        d_keys_input, d_keys_output, d_values_input, d_values_output, size,
        0, sizeof(Key) * 8,
        acc_view
    );
}

template<benchmark_kinds benchmark_kind, class T>
void run_benchmark(benchmark::State& state, hc::accelerator_view acc_view, size_t size)
{
    using key_type = T;
    using value_type = T;

    // Generate data
    std::vector<key_type> keys_input;
    if(std::is_floating_point<key_type>::value)
    {
        keys_input = get_random_data<key_type>(size, (key_type)-1000, (key_type)+1000, size);
    }
    else
    {
        keys_input = get_random_data<key_type>(
            size,
            std::numeric_limits<key_type>::min(),
            std::numeric_limits<key_type>::max(),
            size
        );
    }

    std::vector<value_type> values_input(size);
    std::iota(values_input.begin(), values_input.end(), 0);

    hc::array<key_type> d_keys_input(hc::extent<1>(size), keys_input.begin(), acc_view);
    hc::array<value_type> d_values_input(hc::extent<1>(size), values_input.begin(), acc_view);
    hc::array<key_type> d_keys_output(size, acc_view);
    hc::array<value_type> d_values_output(size, acc_view);
    acc_view.wait();

    size_t temp_storage_size_bytes = 0;

    run_radix_sort<benchmark_kind>(
        nullptr,
        temp_storage_size_bytes,
        d_keys_input.accelerator_pointer(),
        d_keys_output.accelerator_pointer(),
        d_values_input.accelerator_pointer(),
        d_values_output.accelerator_pointer(),
        size,
        acc_view
    );
    acc_view.wait();

    // allocate temporary storage
    hc::array<char> d_temp_storage(temp_storage_size_bytes, acc_view);
    acc_view.wait();

    // Warm-up
    for(size_t i = 0; i < 10; i++)
    {
        run_radix_sort<benchmark_kind>(
            d_temp_storage.accelerator_pointer(),
            temp_storage_size_bytes,
            d_keys_input.accelerator_pointer(),
            d_keys_output.accelerator_pointer(),
            d_values_input.accelerator_pointer(),
            d_values_output.accelerator_pointer(),
            size,
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
            run_radix_sort<benchmark_kind>(
                d_temp_storage.accelerator_pointer(),
                temp_storage_size_bytes,
                d_keys_input.accelerator_pointer(),
                d_keys_output.accelerator_pointer(),
                d_values_input.accelerator_pointer(),
                d_values_output.accelerator_pointer(),
                size,
                acc_view
            );
        }
        acc_view.wait();

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }
    state.SetBytesProcessed(state.iterations() * batch_size * size * sizeof(key_type));
    state.SetItemsProcessed(state.iterations() * batch_size * size);
}

#define CREATE_BENCHMARK(T) \
benchmark::RegisterBenchmark( \
    (std::string("radix_") + name + "<" #T ">").c_str(), \
    run_benchmark<benchmark_kind, T>, \
    acc_view, size \
)

template<benchmark_kinds benchmark_kind>
void add_benchmarks(const std::string& name,
                    std::vector<benchmark::internal::Benchmark*>& benchmarks,
                    hc::accelerator_view acc_view,
                    size_t size)
{
    std::vector<benchmark::internal::Benchmark*> bs =
    {
        CREATE_BENCHMARK(unsigned int),
        CREATE_BENCHMARK(int),

        CREATE_BENCHMARK(unsigned long long),
        CREATE_BENCHMARK(long long),

        CREATE_BENCHMARK(char),
        CREATE_BENCHMARK(short),

        CREATE_BENCHMARK(float),
        CREATE_BENCHMARK(double),
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
    add_benchmarks<benchmark_kinds::sort_keys>("sort_keys", benchmarks, acc_view, size);
    add_benchmarks<benchmark_kinds::sort_keys_desc>("sort_keys_desc", benchmarks, acc_view, size);
    add_benchmarks<benchmark_kinds::sort_pairs>("sort_pairs", benchmarks, acc_view, size);
    add_benchmarks<benchmark_kinds::sort_pairs_desc>("sort_pairs_desc", benchmarks, acc_view, size);

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
