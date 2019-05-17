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

const unsigned int batch_size = 10;
const unsigned int warmup_size = 5;

template<class T>
void run_benchmark(benchmark::State& state, size_t desired_segments, hipStream_t stream, size_t size)
{
    using offset_type = int;
    using value_type = T;

    // Generate data
    const unsigned int seed = 123;
    std::default_random_engine gen(seed);

    const double avg_segment_length = static_cast<double>(size) / desired_segments;
    std::uniform_real_distribution<double> segment_length_dis(0, avg_segment_length * 2);

    std::vector<offset_type> offsets;
    unsigned int segments_count = 0;
    size_t offset = 0;
    while(offset < size)
    {
        const size_t segment_length = std::round(segment_length_dis(gen));
        offsets.push_back(offset);
        segments_count++;
        offset += segment_length;
    }
    offsets.push_back(size);

    std::vector<value_type> values_input(size);
    std::iota(values_input.begin(), values_input.end(), 0);

    offset_type * d_offsets;
    HIP_CHECK(hipMalloc(&d_offsets, (segments_count + 1) * sizeof(offset_type)));
    HIP_CHECK(
        hipMemcpy(
            d_offsets, offsets.data(),
            (segments_count + 1) * sizeof(offset_type),
            hipMemcpyHostToDevice
        )
    );

    value_type * d_values_input;
    HIP_CHECK(hipMalloc(&d_values_input, size * sizeof(value_type)));
    HIP_CHECK(
        hipMemcpy(
            d_values_input, values_input.data(),
            size * sizeof(value_type),
            hipMemcpyHostToDevice
        )
    );

    value_type * d_aggregates_output;
    HIP_CHECK(hipMalloc(&d_aggregates_output, segments_count * sizeof(value_type)));

    rocprim::plus<value_type> reduce_op;
    value_type init(0);

    void * d_temporary_storage = nullptr;
    size_t temporary_storage_bytes = 0;

    HIP_CHECK(
        rp::segmented_reduce(
            d_temporary_storage, temporary_storage_bytes,
            d_values_input, d_aggregates_output,
            segments_count,
            d_offsets, d_offsets + 1,
            reduce_op, init,
            stream
        )
    );

    HIP_CHECK(hipMalloc(&d_temporary_storage, temporary_storage_bytes));
    HIP_CHECK(hipDeviceSynchronize());

    // Warm-up
    for(size_t i = 0; i < warmup_size; i++)
    {
        HIP_CHECK(
            rp::segmented_reduce(
                d_temporary_storage, temporary_storage_bytes,
                d_values_input, d_aggregates_output,
                segments_count,
                d_offsets, d_offsets + 1,
                reduce_op, init,
                stream
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
                rp::segmented_reduce(
                    d_temporary_storage, temporary_storage_bytes,
                    d_values_input, d_aggregates_output,
                    segments_count,
                    d_offsets, d_offsets + 1,
                    reduce_op, init,
                    stream
                )
            );
        }
        HIP_CHECK(hipStreamSynchronize(stream));

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }
    state.SetBytesProcessed(state.iterations() * batch_size * size * sizeof(value_type));
    state.SetItemsProcessed(state.iterations() * batch_size * size);

    HIP_CHECK(hipFree(d_temporary_storage));
    HIP_CHECK(hipFree(d_offsets));
    HIP_CHECK(hipFree(d_values_input));
    HIP_CHECK(hipFree(d_aggregates_output));
}

#define CREATE_BENCHMARK(T, SEGMENTS) \
benchmark::RegisterBenchmark( \
    (std::string("segmented_reduce") + "<" #T ">" + \
        "(~" + std::to_string(SEGMENTS) + " segments)" \
    ).c_str(), \
    run_benchmark<T>, \
    SEGMENTS, stream, size \
)

#define BENCHMARK_TYPE(type) \
    CREATE_BENCHMARK(type, 1), \
    CREATE_BENCHMARK(type, 10), \
    CREATE_BENCHMARK(type, 100), \
    CREATE_BENCHMARK(type, 1000), \
    CREATE_BENCHMARK(type, 10000)

void add_benchmarks(std::vector<benchmark::internal::Benchmark*>& benchmarks,
                    hipStream_t stream,
                    size_t size)
{
    using custom_float2 = custom_type<float, float>;
    using custom_double2 = custom_type<double, double>;

    std::vector<benchmark::internal::Benchmark*> bs =
    {
        BENCHMARK_TYPE(float),
        BENCHMARK_TYPE(double),
        BENCHMARK_TYPE(int8_t),
        BENCHMARK_TYPE(uint8_t),
        BENCHMARK_TYPE(rocprim::half),
        BENCHMARK_TYPE(int),
        BENCHMARK_TYPE(custom_float2),
        BENCHMARK_TYPE(custom_double2),
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
    add_benchmarks(benchmarks, stream, size);

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
