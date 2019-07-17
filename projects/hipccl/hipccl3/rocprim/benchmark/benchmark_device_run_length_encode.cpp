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
#include <string>
#include <limits>

// Google Benchmark
#include "benchmark/benchmark.h"
// CmdParser
#include "cmdparser.hpp"
#include "benchmark_utils.hpp"

// HIP API
#include <hip/hip_runtime.h>

// rocPRIM
#include <rocprim/rocprim.hpp>

#define HIP_CHECK(condition)         \
  {                                   \
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

template<class T>
void run_encode_benchmark(benchmark::State& state, size_t max_length, hipStream_t stream, size_t size)
{
    using key_type = T;
    using count_type = unsigned int;

    // Generate data
    std::vector<key_type> input(size);

    unsigned int runs_count = 0;
    std::vector<size_t> key_counts = get_random_data<size_t>(100000, 1, max_length);
    size_t offset = 0;
    while(offset < size)
    {
        const size_t key_count = key_counts[runs_count % key_counts.size()];
        const size_t end = std::min(size, offset + key_count);
        for(size_t i = offset; i < end; i++)
        {
            input[i] = runs_count;
        }

        runs_count++;
        offset += key_count;
    }

    key_type * d_input;
    HIP_CHECK(hipMalloc(&d_input, size * sizeof(key_type)));
    HIP_CHECK(
        hipMemcpy(
            d_input, input.data(),
            size * sizeof(key_type),
            hipMemcpyHostToDevice
        )
    );

    key_type * d_unique_output;
    count_type * d_counts_output;
    count_type * d_runs_count_output;
    HIP_CHECK(hipMalloc(&d_unique_output, runs_count * sizeof(key_type)));
    HIP_CHECK(hipMalloc(&d_counts_output, runs_count * sizeof(count_type)));
    HIP_CHECK(hipMalloc(&d_runs_count_output, sizeof(count_type)));

    void * d_temporary_storage = nullptr;
    size_t temporary_storage_bytes = 0;

    HIP_CHECK(
        rp::run_length_encode(
            nullptr, temporary_storage_bytes,
            d_input, size,
            d_unique_output, d_counts_output, d_runs_count_output,
            stream, false
        )
    );

    HIP_CHECK(hipMalloc(&d_temporary_storage, temporary_storage_bytes));
    HIP_CHECK(hipDeviceSynchronize());

    // Warm-up
    for(size_t i = 0; i < 10; i++)
    {
        HIP_CHECK(
            rp::run_length_encode(
                d_temporary_storage, temporary_storage_bytes,
                d_input, size,
                d_unique_output, d_counts_output, d_runs_count_output,
                stream, false
            )
        );
    }
    HIP_CHECK(hipDeviceSynchronize());

    const unsigned int batch_size = 10;
    for (auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();

        for(size_t i = 0; i < batch_size; i++)
        {
            rp::run_length_encode(
                d_temporary_storage, temporary_storage_bytes,
                d_input, size,
                d_unique_output, d_counts_output, d_runs_count_output,
                stream, false
            );
        }
        HIP_CHECK(hipStreamSynchronize(stream));

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }
    state.SetBytesProcessed(state.iterations() * batch_size * size * sizeof(key_type));
    state.SetItemsProcessed(state.iterations() * batch_size * size);

    HIP_CHECK(hipFree(d_temporary_storage));
    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_unique_output));
    HIP_CHECK(hipFree(d_counts_output));
    HIP_CHECK(hipFree(d_runs_count_output));
}

template<class T>
void run_non_trivial_runs_benchmark(benchmark::State& state, size_t max_length, hipStream_t stream, size_t size)
{
    using key_type = T;
    using offset_type = unsigned int;
    using count_type = unsigned int;

    // Generate data
    std::vector<key_type> input(size);

    unsigned int runs_count = 0;
    std::vector<size_t> key_counts = get_random_data<size_t>(100000, 1, max_length);
    size_t offset = 0;
    while(offset < size)
    {
        const size_t key_count = key_counts[runs_count % key_counts.size()];
        const size_t end = std::min(size, offset + key_count);
        for(size_t i = offset; i < end; i++)
        {
            input[i] = runs_count;
        }

        runs_count++;
        offset += key_count;
    }

    key_type * d_input;
    HIP_CHECK(hipMalloc(&d_input, size * sizeof(key_type)));
    HIP_CHECK(
        hipMemcpy(
            d_input, input.data(),
            size * sizeof(key_type),
            hipMemcpyHostToDevice
        )
    );

    offset_type * d_offsets_output;
    count_type * d_counts_output;
    count_type * d_runs_count_output;
    HIP_CHECK(hipMalloc(&d_offsets_output, runs_count * sizeof(offset_type)));
    HIP_CHECK(hipMalloc(&d_counts_output, runs_count * sizeof(count_type)));
    HIP_CHECK(hipMalloc(&d_runs_count_output, sizeof(count_type)));

    void * d_temporary_storage = nullptr;
    size_t temporary_storage_bytes = 0;

    HIP_CHECK(
        rp::run_length_encode_non_trivial_runs(
            nullptr, temporary_storage_bytes,
            d_input, size,
            d_offsets_output, d_counts_output, d_runs_count_output,
            stream, false
        )
    );

    HIP_CHECK(hipMalloc(&d_temporary_storage, temporary_storage_bytes));
    HIP_CHECK(hipDeviceSynchronize());

    // Warm-up
    for(size_t i = 0; i < 10; i++)
    {
        HIP_CHECK(
            rp::run_length_encode_non_trivial_runs(
                d_temporary_storage, temporary_storage_bytes,
                d_input, size,
                d_offsets_output, d_counts_output, d_runs_count_output,
                stream, false
            )
        );
    }
    HIP_CHECK(hipDeviceSynchronize());

    const unsigned int batch_size = 10;
    for (auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();

        for(size_t i = 0; i < batch_size; i++)
        {
            rp::run_length_encode_non_trivial_runs(
                d_temporary_storage, temporary_storage_bytes,
                d_input, size,
                d_offsets_output, d_counts_output, d_runs_count_output,
                stream, false
            );
        }
        HIP_CHECK(hipStreamSynchronize(stream));

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }
    state.SetBytesProcessed(state.iterations() * batch_size * size * sizeof(key_type));
    state.SetItemsProcessed(state.iterations() * batch_size * size);

    HIP_CHECK(hipFree(d_temporary_storage));
    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_offsets_output));
    HIP_CHECK(hipFree(d_counts_output));
    HIP_CHECK(hipFree(d_runs_count_output));
}

#define CREATE_ENCODE_BENCHMARK(T) \
benchmark::RegisterBenchmark( \
    (std::string("run_length_encode") + "<" #T ">" + \
        "([1, " + std::to_string(max_length) + "])" \
    ).c_str(), \
    run_encode_benchmark<T>, \
    max_length, stream, size \
)

void add_encode_benchmarks(size_t max_length,
                           std::vector<benchmark::internal::Benchmark*>& benchmarks,
                           hipStream_t stream,
                           size_t size)
{
    using custom_float2 = custom_type<float, float>;
    using custom_double2 = custom_type<double, double>;

    std::vector<benchmark::internal::Benchmark*> bs =
    {
        CREATE_ENCODE_BENCHMARK(int),
        CREATE_ENCODE_BENCHMARK(long long),

        CREATE_ENCODE_BENCHMARK(int8_t),
        CREATE_ENCODE_BENCHMARK(uint8_t),
        CREATE_ENCODE_BENCHMARK(rocprim::half),

        CREATE_ENCODE_BENCHMARK(custom_float2),
        CREATE_ENCODE_BENCHMARK(custom_double2),
    };

    benchmarks.insert(benchmarks.end(), bs.begin(), bs.end());
}

#define CREATE_NON_TRIVIAL_RUNS_BENCHMARK(T) \
benchmark::RegisterBenchmark( \
    (std::string("run_length_encode_non_trivial_runs") + "<" #T ">" + \
        "([1, " + std::to_string(max_length) + "])" \
    ).c_str(), \
    run_non_trivial_runs_benchmark<T>, \
    max_length, stream, size \
)

void add_non_trivial_runs_benchmarks(size_t max_length,
                                     std::vector<benchmark::internal::Benchmark*>& benchmarks,
                                     hipStream_t stream,
                                     size_t size)
{
    using custom_float2 = custom_type<float, float>;
    using custom_double2 = custom_type<double, double>;

    std::vector<benchmark::internal::Benchmark*> bs =
    {
        CREATE_NON_TRIVIAL_RUNS_BENCHMARK(int),
        CREATE_NON_TRIVIAL_RUNS_BENCHMARK(long long),

        CREATE_NON_TRIVIAL_RUNS_BENCHMARK(int8_t),
        CREATE_NON_TRIVIAL_RUNS_BENCHMARK(uint8_t),
        CREATE_NON_TRIVIAL_RUNS_BENCHMARK(rocprim::half),

        CREATE_NON_TRIVIAL_RUNS_BENCHMARK(custom_float2),
        CREATE_NON_TRIVIAL_RUNS_BENCHMARK(custom_double2),
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

    // HIP
    hipStream_t stream = 0; // default
    hipDeviceProp_t devProp;
    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    HIP_CHECK(hipGetDeviceProperties(&devProp, device_id));
    std::cout << "[HIP] Device name: " << devProp.name << std::endl;

    // Add benchmarks
    std::vector<benchmark::internal::Benchmark*> benchmarks;
    add_encode_benchmarks(1000, benchmarks, stream, size);
    add_encode_benchmarks(10, benchmarks, stream, size);
    add_non_trivial_runs_benchmarks(1000, benchmarks, stream, size);
    add_non_trivial_runs_benchmarks(10, benchmarks, stream, size);

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
