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

// Google Benchmark
#include "benchmark/benchmark.h"
// CmdParser
#include "cmdparser.hpp"
// HIP API
#include <hip/hip_runtime.h>

// rocPRIM
#include <rocprim/rocprim.hpp>

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
const size_t DEFAULT_N = 1024 * 1024 * 128;
#endif

template<
    bool Exclusive,
    class T,
    class BinaryFunction
>
auto run_device_scan(void * temporary_storage,
                     size_t& storage_size,
                     T * input,
                     T * output,
                     const T initial_value,
                     const size_t input_size,
                     BinaryFunction scan_op,
                     const hipStream_t stream,
                     const bool debug = false)
    -> typename std::enable_if<Exclusive, hipError_t>::type
{
    return rocprim::exclusive_scan(
        temporary_storage, storage_size,
        input, output, initial_value, input_size,
        scan_op, stream, debug
    );
}

template<
    bool Exclusive,
    class T,
    class BinaryFunction
>
auto run_device_scan(void * temporary_storage,
                     size_t& storage_size,
                     T * input,
                     T * output,
                     const T initial_value,
                     const size_t input_size,
                     BinaryFunction scan_op,
                     const hipStream_t stream,
                     const bool debug = false)
    -> typename std::enable_if<!Exclusive, hipError_t>::type
{
    (void) initial_value;
    return rocprim::inclusive_scan(
        temporary_storage, storage_size,
        input, output, input_size,
        scan_op, stream, debug
    );
}

template<
    bool Exclusive,
    class T,
    class BinaryFunction
>
void run_benchmark(benchmark::State& state,
                   size_t size,
                   const hipStream_t stream,
                   BinaryFunction scan_op)
{
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
    T initial_value = get_random_value<T>((T)-1000, (T)+1000);
    T * d_input;
    T * d_output;
    HIP_CHECK(hipMalloc(&d_input, size * sizeof(T)));
    HIP_CHECK(hipMalloc(&d_output, size * sizeof(T)));
    HIP_CHECK(
        hipMemcpy(
            d_input, input.data(),
            size * sizeof(T),
            hipMemcpyHostToDevice
        )
    );
    HIP_CHECK(hipDeviceSynchronize());

    // Allocate temporary storage memory
    size_t temp_storage_size_bytes;
    void * d_temp_storage = nullptr;
    // Get size of d_temp_storage
    HIP_CHECK(
        run_device_scan<Exclusive>(
            d_temp_storage, temp_storage_size_bytes,
            d_input, d_output, initial_value, size,
            scan_op, stream
        )
    );
    HIP_CHECK(hipMalloc(&d_temp_storage,temp_storage_size_bytes));
    HIP_CHECK(hipDeviceSynchronize());

    // Warm-up
    for(size_t i = 0; i < 10; i++)
    {
        HIP_CHECK(
            run_device_scan<Exclusive>(
                d_temp_storage, temp_storage_size_bytes,
                d_input, d_output, initial_value, size,
                scan_op, stream
            )
        );
    }
    HIP_CHECK(hipDeviceSynchronize());

    const unsigned int batch_size = 10;
    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();
        for(size_t i = 0; i < batch_size; i++)
        {
            HIP_CHECK(
                run_device_scan<Exclusive>(
                    d_temp_storage, temp_storage_size_bytes,
                    d_input, d_output, initial_value, size,
                    scan_op, stream
                )
            );
        }
        HIP_CHECK(hipStreamSynchronize(stream));

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }
    state.SetBytesProcessed(state.iterations() * batch_size * size * sizeof(T));
    state.SetItemsProcessed(state.iterations() * batch_size * size);

    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));
    HIP_CHECK(hipFree(d_temp_storage));
}

#define CREATE_INCLUSIVE_BENCHMARK(T, SCAN_OP) \
benchmark::RegisterBenchmark( \
    ("inclusive_scan<" #T "," #SCAN_OP ">"), \
    run_benchmark<false, T, SCAN_OP>, size, stream, SCAN_OP() \
)

#define CREATE_EXCLUSIVE_BENCHMARK(T, SCAN_OP) \
benchmark::RegisterBenchmark( \
    ("exclusive_scan<" #T "," #SCAN_OP ">"), \
    run_benchmark<true, T, SCAN_OP>, size, stream, SCAN_OP() \
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

    using custom_double2 = custom_type<double, double>;
    using custom_int2 = custom_type<int, int>;

    // Add benchmarks
    std::vector<benchmark::internal::Benchmark*> benchmarks =
    {
        CREATE_INCLUSIVE_BENCHMARK(int, rocprim::plus<int>),
        CREATE_EXCLUSIVE_BENCHMARK(int, rocprim::plus<int>),

        CREATE_INCLUSIVE_BENCHMARK(int8_t, rocprim::plus<int8_t>),
        CREATE_EXCLUSIVE_BENCHMARK(int8_t, rocprim::plus<int8_t>),

        CREATE_INCLUSIVE_BENCHMARK(uint8_t, rocprim::plus<uint8_t>),
        CREATE_EXCLUSIVE_BENCHMARK(uint8_t, rocprim::plus<uint8_t>),

        CREATE_INCLUSIVE_BENCHMARK(rocprim::half, rocprim::plus<rocprim::half>),
        CREATE_EXCLUSIVE_BENCHMARK(rocprim::half, rocprim::plus<rocprim::half>),

        CREATE_INCLUSIVE_BENCHMARK(float, rocprim::plus<float>),
        CREATE_EXCLUSIVE_BENCHMARK(float, rocprim::plus<float>),

        CREATE_INCLUSIVE_BENCHMARK(double, rocprim::plus<double>),
        CREATE_EXCLUSIVE_BENCHMARK(double, rocprim::plus<double>),

        CREATE_INCLUSIVE_BENCHMARK(long long, rocprim::plus<long long>),
        CREATE_EXCLUSIVE_BENCHMARK(long long, rocprim::plus<long long>),

        CREATE_INCLUSIVE_BENCHMARK(custom_double2, rocprim::plus<custom_double2>),
        CREATE_EXCLUSIVE_BENCHMARK(custom_double2, rocprim::plus<custom_double2>),
        CREATE_INCLUSIVE_BENCHMARK(custom_int2, rocprim::plus<custom_int2>),
        CREATE_EXCLUSIVE_BENCHMARK(custom_int2, rocprim::plus<custom_int2>)
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
