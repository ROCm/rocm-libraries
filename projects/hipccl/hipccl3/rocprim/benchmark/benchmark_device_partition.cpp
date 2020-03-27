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
#include <locale>
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

#include <rocprim/rocprim.hpp>

#define HIP_CHECK(condition)             \
    {                                    \
        hipError_t error = condition;    \
        if(error != hipSuccess){         \
            std::cout << "HIP error: " << error << " line: " << __LINE__ << std::endl; \
            exit(error); \
        } \
    }

#ifndef DEFAULT_N
const size_t DEFAULT_N = 1024 * 1024 * 32;
#endif

template<class T, class FlagType>
void run_flagged_benchmark(benchmark::State& state,
                           size_t size,
                           const hipStream_t stream,
                           float true_probability)
{
    size = (size * sizeof(int)) / sizeof(T);

    std::vector<T> input;
    std::vector<FlagType> flags = get_random_data01<FlagType>(size, true_probability);
    if(std::is_floating_point<T>::value)
    {
        input = get_random_data<T>(size, T(-1000), T(1000));
    }
    else
    {
        input = get_random_data<T>(
            size,
            std::numeric_limits<T>::min(),
            std::numeric_limits<T>::max()
        );
    }

    T * d_input;
    FlagType * d_flags;
    T * d_output;
    unsigned int * d_selected_count_output;
    HIP_CHECK(hipMalloc(&d_input, input.size() * sizeof(T)));
    HIP_CHECK(hipMalloc(&d_flags, flags.size() * sizeof(FlagType)));
    HIP_CHECK(hipMalloc(&d_output, input.size() * sizeof(T)));
    HIP_CHECK(hipMalloc(&d_selected_count_output, sizeof(unsigned int)));
    HIP_CHECK(
        hipMemcpy(
            d_input, input.data(),
            input.size() * sizeof(T),
            hipMemcpyHostToDevice
        )
    );
    HIP_CHECK(
        hipMemcpy(
            d_flags, flags.data(),
            flags.size() * sizeof(FlagType),
            hipMemcpyHostToDevice
        )
    );
    HIP_CHECK(hipDeviceSynchronize());
    // Allocate temporary storage memory
    size_t temp_storage_size_bytes;

    // Get size of d_temp_storage
    rocprim::partition(
        nullptr,
        temp_storage_size_bytes,
        d_input,
        d_flags,
        d_output,
        d_selected_count_output,
        input.size(),
        stream
    );
    HIP_CHECK(hipDeviceSynchronize());

    // allocate temporary storage
    void * d_temp_storage = nullptr;
    HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_size_bytes));
    HIP_CHECK(hipDeviceSynchronize());

    // Warm-up
    for(size_t i = 0; i < 10; i++)
    {
        rocprim::partition(
            d_temp_storage,
            temp_storage_size_bytes,
            d_input,
            d_flags,
            d_output,
            d_selected_count_output,
            input.size(),
            stream
        );
    }
    HIP_CHECK(hipDeviceSynchronize());

    const unsigned int batch_size = 10;
    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();
        for(size_t i = 0; i < batch_size; i++)
        {
            rocprim::partition(
                d_temp_storage,
                temp_storage_size_bytes,
                d_input,
                d_flags,
                d_output,
                d_selected_count_output,
                input.size(),
                stream
            );
        }
        HIP_CHECK(hipDeviceSynchronize());

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }
    state.SetBytesProcessed(state.iterations() * batch_size * size * sizeof(T));
    state.SetItemsProcessed(state.iterations() * batch_size * size);

    hipFree(d_input);
    hipFree(d_flags);
    hipFree(d_output);
    hipFree(d_selected_count_output);
    hipFree(d_temp_storage);
}

template<class T>
void run_if_benchmark(benchmark::State& state,
                      size_t size,
                      const hipStream_t stream,
                      float true_probability)
{
    auto select_op = [true_probability] __device__ (const T& value) -> bool
    {
        if(value < T(10000 * true_probability)) return true;
        return false;
    };

    std::vector<T> input = get_random_data<T>(size, T(0), T(10000));
    T * d_input;
    T * d_output;
    unsigned int * d_selected_count_output;
    HIP_CHECK(hipMalloc(&d_input, input.size() * sizeof(T)));
    HIP_CHECK(hipMalloc(&d_output, input.size() * sizeof(T)));
    HIP_CHECK(hipMalloc(&d_selected_count_output, sizeof(unsigned int)));
    HIP_CHECK(
        hipMemcpy(
            d_input, input.data(),
            input.size() * sizeof(T),
            hipMemcpyHostToDevice
        )
    );
    HIP_CHECK(hipDeviceSynchronize());
    // Allocate temporary storage memory
    size_t temp_storage_size_bytes;

    // Get size of d_temp_storage
    rocprim::partition(
        nullptr,
        temp_storage_size_bytes,
        d_input,
        d_output,
        d_selected_count_output,
        input.size(),
        select_op,
        stream
    );
    HIP_CHECK(hipDeviceSynchronize());

    // allocate temporary storage
    void * d_temp_storage = nullptr;
    HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_size_bytes));
    HIP_CHECK(hipDeviceSynchronize());

    // Warm-up
    for(size_t i = 0; i < 10; i++)
    {
        rocprim::partition(
            d_temp_storage,
            temp_storage_size_bytes,
            d_input,
            d_output,
            d_selected_count_output,
            input.size(),
            select_op,
            stream
        );
    }
    HIP_CHECK(hipDeviceSynchronize());

    const unsigned int batch_size = 10;
    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();
        for(size_t i = 0; i < batch_size; i++)
        {
            rocprim::partition(
                d_temp_storage,
                temp_storage_size_bytes,
                d_input,
                d_output,
                d_selected_count_output,
                input.size(),
                select_op,
                stream
            );
        }
        HIP_CHECK(hipDeviceSynchronize());

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }
    state.SetBytesProcessed(state.iterations() * batch_size * size * sizeof(T));
    state.SetItemsProcessed(state.iterations() * batch_size * size);

    hipFree(d_input);
    hipFree(d_output);
    hipFree(d_selected_count_output);
    hipFree(d_temp_storage);
}

#define CREATE_PARTITION_FLAGGED_BENCHMARK(T, F, p) \
benchmark::RegisterBenchmark( \
    ("partition(flags)<" #T "," #F ", "#T", unsigned int>(p = " #p")"), \
    run_flagged_benchmark<T, F>, size, stream, p \
)

#define CREATE_PARTITION_IF_BENCHMARK(T, p) \
benchmark::RegisterBenchmark( \
    ("partition(if)<" #T ", "#T", unsigned int>(p = " #p")"), \
    run_if_benchmark<T>, size, stream, p \
)

#define BENCHMARK_FLAGGED_TYPE(type, value) \
    CREATE_PARTITION_FLAGGED_BENCHMARK(type, value, 0.05f), \
    CREATE_PARTITION_FLAGGED_BENCHMARK(type, value, 0.25f), \
    CREATE_PARTITION_FLAGGED_BENCHMARK(type, value, 0.5f), \
    CREATE_PARTITION_FLAGGED_BENCHMARK(type, value, 0.75f)

#define BENCHMARK_IF_TYPE(type) \
    CREATE_PARTITION_IF_BENCHMARK(type, 0.05f), \
    CREATE_PARTITION_IF_BENCHMARK(type, 0.25f), \
    CREATE_PARTITION_IF_BENCHMARK(type, 0.5f), \
    CREATE_PARTITION_IF_BENCHMARK(type, 0.75f)

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

    using custom_double2 = custom_type<double, double>;
    using custom_int_double = custom_type<int, double>;

    // Add benchmarks
    std::vector<benchmark::internal::Benchmark*> benchmarks =
    {
        BENCHMARK_FLAGGED_TYPE(int, unsigned char),
        BENCHMARK_FLAGGED_TYPE(float, unsigned char),
        BENCHMARK_FLAGGED_TYPE(double, unsigned char),
        BENCHMARK_FLAGGED_TYPE(uint8_t, uint8_t),
        BENCHMARK_FLAGGED_TYPE(int8_t, int8_t),
        BENCHMARK_FLAGGED_TYPE(rocprim::half, int8_t),
        BENCHMARK_FLAGGED_TYPE(custom_double2, unsigned char),

        BENCHMARK_IF_TYPE(int),
        BENCHMARK_IF_TYPE(float),
        BENCHMARK_IF_TYPE(double),
        BENCHMARK_IF_TYPE(uint8_t),
        BENCHMARK_IF_TYPE(int8_t),
        BENCHMARK_IF_TYPE(rocprim::half),
        BENCHMARK_IF_TYPE(custom_int_double)
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
