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

// HIP API
#include <hip/hip_runtime.h>

// rocPRIM HIP API
#include <device/device_transform_hip.hpp>

// CmdParser
#include "cmdparser.hpp"
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

template<class T>
struct transform
{
    constexpr T operator()(const T& a) const [[hc]] [[cpu]]
    {
        return a + 5;
    }
};

template<class T>
struct transform2
{
    constexpr T operator()(const T& a, const T& b) const [[hc]] [[cpu]]
    {
        return a + b;
    }
};

template<
    class T,
    class BinaryFunction
>
void run_benchmark(benchmark::State& state,
                   size_t size,
                   const hipStream_t stream,
                   BinaryFunction transform_op)
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

    // Warm-up
    for(size_t i = 0; i < 10; i++)
    {
        HIP_CHECK(
            rocprim::device_transform(
                d_input, d_output, size,
                transform_op, stream
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
                rocprim::device_transform(
                    d_input, d_output, size,
                    transform_op, stream
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
}

template<
    class T,
    class BinaryFunction
>
void run_benchmark2(benchmark::State& state,
                   size_t size,
                   const hipStream_t stream,
                   BinaryFunction transform_op)
{
    std::vector<T> input;
    std::vector<T> output;
    if(std::is_floating_point<T>::value)
    {
        input = get_random_data<T>(size, (T)-1000, (T)+1000);
        output = get_random_data<T>(size, (T)-1000, (T)+1000);
    }
    else
    {
        input = get_random_data<T>(
            size,
            std::numeric_limits<T>::min(),
            std::numeric_limits<T>::max()
        );
        output = get_random_data<T>(
            size,
            std::numeric_limits<T>::min(),
            std::numeric_limits<T>::max()
        );
    }

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
    HIP_CHECK(
        hipMemcpy(
            d_output, output.data(),
            size * sizeof(T),
            hipMemcpyHostToDevice
        )
    );
    HIP_CHECK(hipDeviceSynchronize());

    // Warm-up
    for(size_t i = 0; i < 10; i++)
    {
        HIP_CHECK(
            rocprim::device_transform(
                d_input, d_output, d_output, size,
                transform_op, stream
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
                rocprim::device_transform(
                    d_input, d_output, d_output, size,
                    transform_op, stream
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
}

#define CREATE_BENCHMARK(T, TRANSFORM_OP) \
benchmark::RegisterBenchmark( \
    ("device_transform<" #T "," #TRANSFORM_OP ">"), \
    run_benchmark<T, TRANSFORM_OP>, size, stream, TRANSFORM_OP() \
)

#define CREATE_BENCHMARK2(T, TRANSFORM_OP) \
benchmark::RegisterBenchmark( \
    ("device_transform<" #T "," #TRANSFORM_OP ">"), \
    run_benchmark2<T, TRANSFORM_OP>, size, stream, TRANSFORM_OP() \
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

    // Add benchmarks
    std::vector<benchmark::internal::Benchmark*> benchmarks =
    {
        CREATE_BENCHMARK(unsigned int, transform<unsigned int>),
        CREATE_BENCHMARK(unsigned long long, transform<unsigned long long>),

        CREATE_BENCHMARK(float, transform<float>),
        CREATE_BENCHMARK(double, transform<double>),
        
        CREATE_BENCHMARK2(unsigned int, transform2<unsigned int>),
        CREATE_BENCHMARK2(unsigned long long, transform2<unsigned long long>),

        CREATE_BENCHMARK2(float, transform2<float>),
        CREATE_BENCHMARK2(double, transform2<double>),
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
