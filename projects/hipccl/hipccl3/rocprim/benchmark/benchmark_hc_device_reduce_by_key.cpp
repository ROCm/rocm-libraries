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

namespace rp = rocprim;

const unsigned int batch_size = 10;
const unsigned int warmup_size = 5;

template<class Key, class Value>
void run_benchmark(benchmark::State& state, size_t max_length, hc::accelerator_view acc_view, size_t size)
{
    using key_type = Key;
    using value_type = Value;

    // Generate data
    std::vector<key_type> keys_input(size);

    unsigned int unique_count = 0;
    std::vector<size_t> key_counts = get_random_data<size_t>(100000, 1, max_length);
    size_t offset = 0;
    while(offset < size)
    {
        const size_t key_count = key_counts[unique_count % key_counts.size()];
        const size_t end = std::min(size, offset + key_count);
        for(size_t i = offset; i < end; i++)
        {
            keys_input[i] = unique_count;
        }

        unique_count++;
        offset += key_count;
    }

    std::vector<value_type> values_input(size);
    std::iota(values_input.begin(), values_input.end(), 0);

    hc::array<key_type> d_keys_input(hc::extent<1>(size), keys_input.begin(), acc_view);
    hc::array<value_type> d_values_input(hc::extent<1>(size), values_input.begin(), acc_view);
    hc::array<key_type> d_unique_output(unique_count, acc_view);
    hc::array<value_type> d_aggregates_output(unique_count, acc_view);
    hc::array<unsigned int> d_unique_count_output(1, acc_view);
    acc_view.wait();

    size_t temp_storage_size_bytes = 0;

    rp::plus<value_type> reduce_op;
    rp::equal_to<key_type> key_compare_op;

    rp::reduce_by_key(
        nullptr,
        temp_storage_size_bytes,
        d_keys_input.accelerator_pointer(),
        d_values_input.accelerator_pointer(),
        size,
        d_unique_output.accelerator_pointer(),
        d_aggregates_output.accelerator_pointer(),
        d_unique_count_output.accelerator_pointer(),
        reduce_op,
        key_compare_op,
        acc_view
    );
    acc_view.wait();

    // allocate temporary storage
    hc::array<char> d_temp_storage(temp_storage_size_bytes, acc_view);
    acc_view.wait();

    // Warm-up
    for(size_t i = 0; i < warmup_size; i++)
    {
        rp::reduce_by_key(
            d_temp_storage.accelerator_pointer(),
            temp_storage_size_bytes,
            d_keys_input.accelerator_pointer(),
            d_values_input.accelerator_pointer(),
            size,
            d_unique_output.accelerator_pointer(),
            d_aggregates_output.accelerator_pointer(),
            d_unique_count_output.accelerator_pointer(),
            reduce_op,
            key_compare_op,
            acc_view
        );
    }
    acc_view.wait();

    for (auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();

        for(size_t i = 0; i < batch_size; i++)
        {
            rp::reduce_by_key(
                d_temp_storage.accelerator_pointer(),
                temp_storage_size_bytes,
                d_keys_input.accelerator_pointer(),
                d_values_input.accelerator_pointer(),
                size,
                d_unique_output.accelerator_pointer(),
                d_aggregates_output.accelerator_pointer(),
                d_unique_count_output.accelerator_pointer(),
                reduce_op,
                key_compare_op,
                acc_view
            );
        }
        acc_view.wait();

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }
    state.SetBytesProcessed(state.iterations() * batch_size * size * (sizeof(key_type) + sizeof(value_type)));
    state.SetItemsProcessed(state.iterations() * batch_size * size);
}

#define CREATE_BENCHMARK(Key, Value) \
benchmark::RegisterBenchmark( \
    (std::string("reduce_by_key") + "<" #Key ", " #Value ">" + \
        "([1, " + std::to_string(max_length) + "])" \
    ).c_str(), \
    run_benchmark<Key, Value>, \
    max_length, acc_view, size \
)

void add_benchmarks(size_t max_length,
                    std::vector<benchmark::internal::Benchmark*>& benchmarks,
                    hc::accelerator_view acc_view,
                    size_t size)
{
    using custom_float2 = custom_type<float, float>;
    using custom_double2 = custom_type<double, double>;

    std::vector<benchmark::internal::Benchmark*> bs =
    {
        CREATE_BENCHMARK(int, float),
        CREATE_BENCHMARK(int, double),
        CREATE_BENCHMARK(int, custom_float2),
        CREATE_BENCHMARK(int, custom_double2),

        CREATE_BENCHMARK(long long, float),
        CREATE_BENCHMARK(long long, double),
        CREATE_BENCHMARK(long long, custom_float2),
        CREATE_BENCHMARK(long long, custom_double2),
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
    add_benchmarks(1000, benchmarks, acc_view, size);
    add_benchmarks(10, benchmarks, acc_view, size);

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
