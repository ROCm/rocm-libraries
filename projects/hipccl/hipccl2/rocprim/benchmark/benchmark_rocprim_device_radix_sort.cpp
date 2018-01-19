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

#include "benchmark_utils.hpp"

// HIP API
#include <hip/hip_runtime.h>
#include <hip/hip_hcc.h>

// rocPRIM
#include <device/device_radix_sort.hpp>

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

namespace rp = rocprim;

template<class Key>
void run_benchmark(benchmark::State& state, hipStream_t stream, size_t size)
{
    std::vector<Key> key_input;
    if(std::is_floating_point<Key>::value)
    {
        key_input = get_random_data<Key>(size, (Key)-1000, (Key)+1000);
    }
    else
    {
        key_input = get_random_data<Key>(
            size,
            std::numeric_limits<Key>::min(),
            std::numeric_limits<Key>::max()
        );
    }
    Key * d_key_input;
    Key * d_key_output;
    HIP_CHECK(hipMalloc(&d_key_input, size * sizeof(Key)));
    HIP_CHECK(hipMalloc(&d_key_output, size * sizeof(Key)));
    HIP_CHECK(
        hipMemcpy(
            d_key_input, key_input.data(),
            size * sizeof(Key),
            hipMemcpyHostToDevice
        )
    );

    void * d_temporary_storage = nullptr;
    size_t temporary_storage_bytes;
    rp::device_radix_sort_keys(
        d_temporary_storage, temporary_storage_bytes,
        d_key_input, d_key_output, size
    );

    HIP_CHECK(hipMalloc(&d_temporary_storage, temporary_storage_bytes));
    HIP_CHECK(hipDeviceSynchronize());

    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();

        rp::device_radix_sort_keys(
            d_temporary_storage, temporary_storage_bytes,
            d_key_input, d_key_output, size,
            0, sizeof(Key) * 8,
            0, false
        );
        HIP_CHECK(hipDeviceSynchronize());

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }
    state.SetBytesProcessed(state.iterations() * size * sizeof(Key));
    state.SetItemsProcessed(state.iterations() * size);

    HIP_CHECK(hipFree(d_temporary_storage));
    HIP_CHECK(hipFree(d_key_input));
    HIP_CHECK(hipFree(d_key_output));
}

#define CREATE_BENCHMARK(Key) \
benchmark::RegisterBenchmark( \
    ("device_radix_sort<" #Key ">"), \
    run_benchmark<Key>, \
    stream, size \
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
        CREATE_BENCHMARK(unsigned int),
        CREATE_BENCHMARK(unsigned long long),

        CREATE_BENCHMARK(char),
        CREATE_BENCHMARK(short),

        CREATE_BENCHMARK(float),
        CREATE_BENCHMARK(double),
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
