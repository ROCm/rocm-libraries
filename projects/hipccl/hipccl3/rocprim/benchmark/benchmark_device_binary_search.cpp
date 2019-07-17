// MIT License
//
// Copyright (c) 2019 Advanced Micro Devices, Inc. All rights reserved.
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

const unsigned int batch_size = 10;
const unsigned int warmup_size = 5;

template<class T>
void run_lower_bound_benchmark(benchmark::State& state, hipStream_t stream,
                               size_t haystack_size, size_t needles_size,
                               bool sorted_needles)
{
    using haystack_type = T;
    using needle_type = T;
    using output_type = size_t;
    using compare_op_type = typename std::conditional<std::is_same<needle_type, rocprim::half>::value, half_less, rocprim::less<needle_type>>::type;

    compare_op_type compare_op;
    // Generate data
    std::vector<haystack_type> haystack(haystack_size);
    std::iota(haystack.begin(), haystack.end(), 0);

    std::vector<needle_type> needles = get_random_data<needle_type>(
        needles_size, needle_type(0), needle_type(haystack_size)
    );
    if(sorted_needles)
    {
        std::sort(needles.begin(), needles.end(), compare_op);
    }

    haystack_type * d_haystack;
    needle_type * d_needles;
    output_type * d_output;
    HIP_CHECK(hipMalloc(&d_haystack, haystack_size * sizeof(haystack_type)));
    HIP_CHECK(hipMalloc(&d_needles, needles_size * sizeof(needle_type)));
    HIP_CHECK(hipMalloc(&d_output, needles_size * sizeof(output_type)));
    HIP_CHECK(
        hipMemcpy(
            d_haystack, haystack.data(),
            haystack_size * sizeof(haystack_type),
            hipMemcpyHostToDevice
        )
    );
    HIP_CHECK(
        hipMemcpy(
            d_needles, needles.data(),
            needles_size * sizeof(needle_type),
            hipMemcpyHostToDevice
        )
    );

    void * d_temporary_storage = nullptr;
    size_t temporary_storage_bytes;
    HIP_CHECK(
        rocprim::lower_bound(
            d_temporary_storage, temporary_storage_bytes,
            d_haystack, d_needles, d_output,
            haystack_size, needles_size,
            compare_op,
            stream
        )
    );

    HIP_CHECK(hipMalloc(&d_temporary_storage, temporary_storage_bytes));

    // Warm-up
    for(size_t i = 0; i < warmup_size; i++)
    {
        HIP_CHECK(
            rocprim::lower_bound(
                d_temporary_storage, temporary_storage_bytes,
                d_haystack, d_needles, d_output,
                haystack_size, needles_size,
                compare_op,
                stream
            )
        );
    }
    HIP_CHECK(hipDeviceSynchronize());

    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();

        for(size_t i = 0; i < batch_size; i++)
        {
            HIP_CHECK(
                rocprim::lower_bound(
                    d_temporary_storage, temporary_storage_bytes,
                    d_haystack, d_needles, d_output,
                    haystack_size, needles_size,
                    compare_op,
                    stream
                )
            );
        }
        HIP_CHECK(hipDeviceSynchronize());

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }
    state.SetBytesProcessed(state.iterations() * batch_size * needles_size * sizeof(needle_type));
    state.SetItemsProcessed(state.iterations() * batch_size * needles_size);

    HIP_CHECK(hipFree(d_temporary_storage));
    HIP_CHECK(hipFree(d_haystack));
    HIP_CHECK(hipFree(d_needles));
    HIP_CHECK(hipFree(d_output));
}

#define CREATE_LOWER_BOUND_BENCHMARK(T, K, SORTED) \
benchmark::RegisterBenchmark( \
    ( \
        std::string("lower_bound") + "<" #T ">(" #K "\% " + \
        (SORTED ? "sorted" : "random") + " needles)" \
    ).c_str(), \
    [=](benchmark::State& state) { run_lower_bound_benchmark<T>(state, stream, size, size * K / 100, SORTED); } \
)

#define BENCHMARK_TYPE(type) \
    CREATE_LOWER_BOUND_BENCHMARK(type, 10, false), \
    CREATE_LOWER_BOUND_BENCHMARK(type, 10, true)

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

    using custom_float2 = custom_type<float, float>;
    using custom_double2 = custom_type<double, double>;

    // Add benchmarks
    std::vector<benchmark::internal::Benchmark*> benchmarks =
    {
        BENCHMARK_TYPE(float),
        BENCHMARK_TYPE(double),
        BENCHMARK_TYPE(int8_t),
        BENCHMARK_TYPE(uint8_t),
        BENCHMARK_TYPE(rocprim::half),
        BENCHMARK_TYPE(custom_float2),
        BENCHMARK_TYPE(custom_double2)
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
