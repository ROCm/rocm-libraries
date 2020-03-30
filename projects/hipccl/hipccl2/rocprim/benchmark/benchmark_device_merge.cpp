// MIT License
//
// Copyright (c) 2017-2019 Advanced Micro Devices, Inc. All rights reserved.
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

// HIP API
#include <hip/hip_runtime.h>

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
const size_t DEFAULT_N = 1024 * 1024 * 32;
#endif

namespace rp = rocprim;

const unsigned int batch_size = 10;
const unsigned int warmup_size = 5;

template<class Key>
void run_merge_keys_benchmark(benchmark::State& state, hipStream_t stream, size_t size)
{
    using key_type = Key;
    using compare_op_type = typename std::conditional<std::is_same<key_type, rocprim::half>::value, half_less, rocprim::less<key_type>>::type;

    const size_t size1 = size / 2;
    const size_t size2 = size - size1;

    compare_op_type compare_op;

    // Generate data
    std::vector<key_type> keys_input1 = get_random_data<key_type>(size1, 0, size);
    std::vector<key_type> keys_input2 = get_random_data<key_type>(size2, 0, size);
    std::sort(keys_input1.begin(), keys_input1.end(), compare_op);
    std::sort(keys_input2.begin(), keys_input2.end(), compare_op);

    key_type * d_keys_input1;
    key_type * d_keys_input2;
    key_type * d_keys_output;
    HIP_CHECK(hipMalloc(&d_keys_input1, size1 * sizeof(key_type)));
    HIP_CHECK(hipMalloc(&d_keys_input2, size2 * sizeof(key_type)));
    HIP_CHECK(hipMalloc(&d_keys_output, size * sizeof(key_type)));
    HIP_CHECK(
        hipMemcpy(
            d_keys_input1, keys_input1.data(),
            size1 * sizeof(key_type),
            hipMemcpyHostToDevice
        )
    );
    HIP_CHECK(
        hipMemcpy(
            d_keys_input2, keys_input2.data(),
            size2 * sizeof(key_type),
            hipMemcpyHostToDevice
        )
    );

    void * d_temporary_storage = nullptr;
    size_t temporary_storage_bytes = 0;
    HIP_CHECK(
        rp::merge(
            d_temporary_storage, temporary_storage_bytes,
            d_keys_input1, d_keys_input2, d_keys_output, size1, size2,
            compare_op, stream, false
        )
    );

    HIP_CHECK(hipMalloc(&d_temporary_storage, temporary_storage_bytes));

    // Warm-up
    for(size_t i = 0; i < warmup_size; i++)
    {
        HIP_CHECK(
            rp::merge(
                d_temporary_storage, temporary_storage_bytes,
                d_keys_input1, d_keys_input2, d_keys_output, size1, size2,
                compare_op, stream, false
            )
        );
    }
    HIP_CHECK(hipDeviceSynchronize());

    for (auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();

        for(size_t i = 0; i < batch_size; i++)
        {
            HIP_CHECK(
                rp::merge(
                    d_temporary_storage, temporary_storage_bytes,
                    d_keys_input1, d_keys_input2, d_keys_output, size1, size2,
                    compare_op, stream, false
                )
            );
        }
        HIP_CHECK(hipDeviceSynchronize());

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }
    state.SetBytesProcessed(state.iterations() * batch_size * size * sizeof(key_type));
    state.SetItemsProcessed(state.iterations() * batch_size * size);

    HIP_CHECK(hipFree(d_temporary_storage));
    HIP_CHECK(hipFree(d_keys_input1));
    HIP_CHECK(hipFree(d_keys_input2));
    HIP_CHECK(hipFree(d_keys_output));
}

template<class Key, class Value>
void run_merge_pairs_benchmark(benchmark::State& state, hipStream_t stream, size_t size)
{
    using key_type = Key;
    using value_type = Value;
    using compare_op_type = typename std::conditional<std::is_same<key_type, rocprim::half>::value, half_less, rocprim::less<key_type>>::type;

    const size_t size1 = size / 2;
    const size_t size2 = size - size1;

    compare_op_type compare_op;

    // Generate data
    std::vector<key_type> keys_input1 = get_random_data<key_type>(size1, 0, size);
    std::vector<key_type> keys_input2 = get_random_data<key_type>(size2, 0, size);
    std::sort(keys_input1.begin(), keys_input1.end(), compare_op);
    std::sort(keys_input2.begin(), keys_input2.end(), compare_op);
    std::vector<value_type> values_input1(size1);
    std::vector<value_type> values_input2(size2);
    std::iota(values_input1.begin(), values_input1.end(), 0);
    std::iota(values_input2.begin(), values_input2.end(), size1);

    key_type * d_keys_input1;
    key_type * d_keys_input2;
    key_type * d_keys_output;
    value_type * d_values_input1;
    value_type * d_values_input2;
    value_type * d_values_output;
    HIP_CHECK(hipMalloc(&d_keys_input1, size1 * sizeof(key_type)));
    HIP_CHECK(hipMalloc(&d_keys_input2, size2 * sizeof(key_type)));
    HIP_CHECK(hipMalloc(&d_keys_output, size * sizeof(key_type)));
    HIP_CHECK(hipMalloc(&d_values_input1, size1 * sizeof(value_type)));
    HIP_CHECK(hipMalloc(&d_values_input2, size2 * sizeof(value_type)));
    HIP_CHECK(hipMalloc(&d_values_output, size * sizeof(value_type)));
    HIP_CHECK(
        hipMemcpy(
            d_keys_input1, keys_input1.data(),
            size1 * sizeof(key_type),
            hipMemcpyHostToDevice
        )
    );
    HIP_CHECK(
        hipMemcpy(
            d_keys_input2, keys_input2.data(),
            size2 * sizeof(key_type),
            hipMemcpyHostToDevice
        )
    );

    void * d_temporary_storage = nullptr;
    size_t temporary_storage_bytes = 0;
    HIP_CHECK(
        rp::merge(
            d_temporary_storage, temporary_storage_bytes,
            d_keys_input1, d_keys_input2, d_keys_output,
            d_values_input1, d_values_input2, d_values_output,
            size1, size2,
            compare_op, stream, false
        )
    );

    HIP_CHECK(hipMalloc(&d_temporary_storage, temporary_storage_bytes));
    HIP_CHECK(hipDeviceSynchronize());

    // Warm-up
    for(size_t i = 0; i < warmup_size; i++)
    {
        HIP_CHECK(
            rp::merge(
                d_temporary_storage, temporary_storage_bytes,
                d_keys_input1, d_keys_input2, d_keys_output,
                d_values_input1, d_values_input2, d_values_output,
                size1, size2,
                compare_op, stream, false
            )
        );
    }
    HIP_CHECK(hipDeviceSynchronize());

    for (auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();

        for(size_t i = 0; i < batch_size; i++)
        {
            HIP_CHECK(
                rp::merge(
                    d_temporary_storage, temporary_storage_bytes,
                    d_keys_input1, d_keys_input2, d_keys_output,
                    d_values_input1, d_values_input2, d_values_output,
                    size1, size2,
                    compare_op, stream, false
                )
            );
        }
        HIP_CHECK(hipDeviceSynchronize());

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }
    state.SetBytesProcessed(state.iterations() * batch_size * size * (sizeof(key_type) + sizeof(value_type)));
    state.SetItemsProcessed(state.iterations() * batch_size * size);

    HIP_CHECK(hipFree(d_temporary_storage));
    HIP_CHECK(hipFree(d_keys_input1));
    HIP_CHECK(hipFree(d_keys_input2));
    HIP_CHECK(hipFree(d_keys_output));
    HIP_CHECK(hipFree(d_values_input1));
    HIP_CHECK(hipFree(d_values_input2));
    HIP_CHECK(hipFree(d_values_output));
}

#define CREATE_MERGE_KEYS_BENCHMARK(Key) \
benchmark::RegisterBenchmark( \
    (std::string("merge") + "<" #Key ">").c_str(), \
    [=](benchmark::State& state) { run_merge_keys_benchmark<Key>(state, stream, size); } \
)

#define CREATE_MERGE_PAIRS_BENCHMARK(Key, Value) \
benchmark::RegisterBenchmark( \
    (std::string("merge") + "<" #Key ", " #Value ">").c_str(), \
    [=](benchmark::State& state) { run_merge_pairs_benchmark<Key, Value>(state, stream, size); } \
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

    // HIP
    hipStream_t stream = 0; // default
    hipDeviceProp_t devProp;
    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    HIP_CHECK(hipGetDeviceProperties(&devProp, device_id));
    std::cout << "[HIP] Device name: " << devProp.name << std::endl;

    using custom_int2 = custom_type<int, int>;
    using custom_double2 = custom_type<double, double>;

    // Add benchmarks
    std::vector<benchmark::internal::Benchmark*> benchmarks =
    {
        CREATE_MERGE_KEYS_BENCHMARK(int),
        CREATE_MERGE_KEYS_BENCHMARK(long long),
        CREATE_MERGE_KEYS_BENCHMARK(int8_t),
        CREATE_MERGE_KEYS_BENCHMARK(uint8_t),
        CREATE_MERGE_KEYS_BENCHMARK(rocprim::half),
        CREATE_MERGE_KEYS_BENCHMARK(short),
        CREATE_MERGE_KEYS_BENCHMARK(custom_int2),
        CREATE_MERGE_KEYS_BENCHMARK(custom_double2),

        CREATE_MERGE_PAIRS_BENCHMARK(int, int),
        CREATE_MERGE_PAIRS_BENCHMARK(long long, long long),
        CREATE_MERGE_PAIRS_BENCHMARK(int8_t, int8_t),
        CREATE_MERGE_PAIRS_BENCHMARK(uint8_t, uint8_t),
        CREATE_MERGE_PAIRS_BENCHMARK(rocprim::half, rocprim::half),
        CREATE_MERGE_PAIRS_BENCHMARK(short, short),
        CREATE_MERGE_PAIRS_BENCHMARK(custom_int2, custom_int2),
        CREATE_MERGE_PAIRS_BENCHMARK(custom_double2, custom_double2),
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
