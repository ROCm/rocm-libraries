// MIT License
//
// Copyright (c) 2017-2022 Advanced Micro Devices, Inc. All rights reserved.
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
#include <string>

// Google Benchmark
#include <benchmark/benchmark.h>

// HIP API
#include <hip/hip_runtime.h>

// rocPRIM
#include <rocprim/rocprim.hpp>

// CmdParser
#include "cmdparser.hpp"

#include "benchmark_device_scan.parallel.hpp"
#include "benchmark_utils.hpp"

#ifndef DEFAULT_N
const size_t DEFAULT_N = 1024 * 1024 * 32;
#endif

#define CREATE_BY_KEY_BENCHMARK(EXCL, T, SCAN_OP, MSL)                     \
    {                                                                      \
        const device_scan_benchmark<true, EXCL, T, SCAN_OP, MSL> instance; \
        REGISTER_BENCHMARK(benchmarks, size, stream, instance);            \
    }

#define CREATE_BENCHMARK(EXCL, T, SCAN_OP)                             \
    {                                                                  \
        const device_scan_benchmark<false, EXCL, T, SCAN_OP> instance; \
        REGISTER_BENCHMARK(benchmarks, size, stream, instance);        \
    }                                                                  \
    CREATE_BY_KEY_BENCHMARK(EXCL, T, SCAN_OP, 1)                       \
    CREATE_BY_KEY_BENCHMARK(EXCL, T, SCAN_OP, 16)                      \
    CREATE_BY_KEY_BENCHMARK(EXCL, T, SCAN_OP, 256)                     \
    CREATE_BY_KEY_BENCHMARK(EXCL, T, SCAN_OP, 4096)                    \
    CREATE_BY_KEY_BENCHMARK(EXCL, T, SCAN_OP, 65536)

int main(int argc, char *argv[])
{
    cli::Parser parser(argc, argv);
    parser.set_optional<size_t>("size", "size", DEFAULT_N, "number of values");
    parser.set_optional<int>("trials", "trials", -1, "number of iterations");
    parser.set_optional<std::string>("name_format",
                                     "name_format",
                                     "human",
                                     "either: json,human,txt");
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
    const size_t size = parser.get<size_t>("size");
    const int trials = parser.get<int>("trials");
    bench_naming::set_format(parser.get<std::string>("name_format"));

    // HIP
    hipStream_t stream = 0; // default

    // Benchmark info
    add_common_benchmark_info();
    benchmark::AddCustomContext("size", std::to_string(size));

    // Add benchmarks
    std::vector<benchmark::internal::Benchmark*> benchmarks = {};
#ifdef BENCHMARK_CONFIG_TUNING
    const int parallel_instance  = parser.get<int>("parallel_instance");
    const int parallel_instances = parser.get<int>("parallel_instances");
    config_autotune_register::register_benchmark_subset(benchmarks,
                                                        parallel_instance,
                                                        parallel_instances,
                                                        size,
                                                        stream);
#else
    using custom_float2  = custom_type<float, float>;
    using custom_double2 = custom_type<double, double>;

    CREATE_BENCHMARK(false, int, rocprim::plus<int>)
    CREATE_BENCHMARK(true, int, rocprim::plus<int>)

    CREATE_BENCHMARK(false, float, rocprim::plus<float>)
    CREATE_BENCHMARK(true, float, rocprim::plus<float>)

    CREATE_BENCHMARK(false, double, rocprim::plus<double>)
    CREATE_BENCHMARK(true, double, rocprim::plus<double>)

    CREATE_BENCHMARK(false, long long, rocprim::plus<long long>)
    CREATE_BENCHMARK(true, long long, rocprim::plus<long long>)

    CREATE_BENCHMARK(false, float2, rocprim::plus<float2>)
    CREATE_BENCHMARK(true, float2, rocprim::plus<float2>)

    CREATE_BENCHMARK(false, custom_float2, rocprim::plus<custom_float2>)
    CREATE_BENCHMARK(true, custom_float2, rocprim::plus<custom_float2>)

    CREATE_BENCHMARK(false, double2, rocprim::plus<double2>)
    CREATE_BENCHMARK(true, double2, rocprim::plus<double2>)

    CREATE_BENCHMARK(false, custom_double2, rocprim::plus<custom_double2>)
    CREATE_BENCHMARK(true, custom_double2, rocprim::plus<custom_double2>)

    CREATE_BENCHMARK(false, int8_t, rocprim::plus<int8_t>)
    CREATE_BENCHMARK(true, int8_t, rocprim::plus<int8_t>)

    CREATE_BENCHMARK(false, uint8_t, rocprim::plus<uint8_t>)
    CREATE_BENCHMARK(true, uint8_t, rocprim::plus<uint8_t>)

    CREATE_BENCHMARK(false, rocprim::half, rocprim::plus<rocprim::half>)
    CREATE_BENCHMARK(true, rocprim::half, rocprim::plus<rocprim::half>)
#endif

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
