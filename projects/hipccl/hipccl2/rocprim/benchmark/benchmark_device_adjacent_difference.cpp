// MIT License
//
// Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include <cstddef>
#include <iostream>
#include <string>

// Google Benchmark
#include <benchmark/benchmark.h>

// HIP API
#include <hip/hip_runtime_api.h>

// rocPRIM
#include <rocprim/detail/various.hpp>
#include <rocprim/device/device_adjacent_difference.hpp>

// CmdParser
#include "cmdparser.hpp"

#include "benchmark_device_adjacent_difference.parallel.hpp"
#include "benchmark_utils.hpp"

#ifndef DEFAULT_N
constexpr std::size_t DEFAULT_N = 1024 * 1024 * 128;
#endif

#define CREATE_BENCHMARK(T, left, in_place)                                     \
    {                                                                           \
        const device_adjacent_difference_benchmark<T, left, in_place> instance; \
        REGISTER_BENCHMARK(benchmarks, size, stream, instance);                 \
    }

// clang-format off
#define CREATE_BENCHMARKS(T)          \
    CREATE_BENCHMARK(T, true,  false) \
    CREATE_BENCHMARK(T, true,  true)  \
    CREATE_BENCHMARK(T, false, false) \
    CREATE_BENCHMARK(T, false, true)
// clang-format on

int main(int argc, char* argv[])
{
    cli::Parser parser(argc, argv);
    parser.set_optional<size_t>("size", "size", DEFAULT_N, "number of values");
    parser.set_optional<int>("trials", "trials", -1, "number of iterations");
#ifdef BENCHMARK_CONFIG_TUNING
    // optionally run an evenly split subset of benchmarks, when making multiple program invocations
    parser.set_optional<int>("parallel_instance",
                             "parallel_instance",
                             0,
                             "parallel instance index");
    parser.set_optional<int>("parallel_instances",
                             "parallel_instances",
                             1,
                             "total parallel instances");
#endif
    parser.run_and_exit_if_error();

    // Parse argv
    benchmark::Initialize(&argc, argv);
    const size_t size   = parser.get<size_t>("size");
    const int    trials = parser.get<int>("trials");

    // HIP
    const hipStream_t stream = 0; // default

    // Benchmark info
    add_common_benchmark_info();
    benchmark::AddCustomContext("size", std::to_string(size));

    std::vector<benchmark::internal::Benchmark*> benchmarks = {};
#ifdef BENCHMARK_CONFIG_TUNING
    const int parallel_instance  = parser.get<int>("parallel_instance");
    const int parallel_instances = parser.get<int>("parallel_instances");
    config_autotune_register::register_benchmark_subset(benchmarks,
                                                        parallel_instance,
                                                        parallel_instances,
                                                        size,
                                                        stream);
    benchmark::AddCustomContext("autotune_config_pattern",
                                device_adjacent_difference_benchmark<>::get_name_pattern().c_str());
#else // BENCHMARK_CONFIG_TUNING
    using custom_float2  = custom_type<float, float>;
    using custom_double2 = custom_type<double, double>;
    // Add benchmarks
    CREATE_BENCHMARKS(int)
    CREATE_BENCHMARKS(std::int64_t)

    CREATE_BENCHMARKS(uint8_t)
    CREATE_BENCHMARKS(rocprim::half)

    CREATE_BENCHMARKS(float)
    CREATE_BENCHMARKS(double)

    CREATE_BENCHMARKS(custom_float2)
    CREATE_BENCHMARKS(custom_double2)
#endif // BENCHMARK_CONFIG_TUNING

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
