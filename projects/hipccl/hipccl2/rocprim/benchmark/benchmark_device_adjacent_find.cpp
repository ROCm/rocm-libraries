// MIT License
//
// Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "benchmark_device_adjacent_find.parallel.hpp"
#include "benchmark_utils.hpp"
#include "cmdparser.hpp"

// gbench
#include <benchmark/benchmark.h>

// HIP
#include <hip/hip_runtime.h>

// C++ Standard Library
#include <string>
#include <vector>

#ifndef DEFAULT_N
const size_t DEFAULT_BYTES = size_t{2} << 30; // 2 GiB
#endif

#define CREATE_BENCHMARK(T, P)                                        \
    {                                                                 \
        const device_adjacent_find_benchmark<T, P> instance;          \
        REGISTER_BENCHMARK(benchmarks, size, seed, stream, instance); \
    }

#define CREATE_ADJACENT_FIND_BENCHMARKS(T) \
    CREATE_BENCHMARK(T, 1)                 \
    CREATE_BENCHMARK(T, 5)                 \
    CREATE_BENCHMARK(T, 9)

int main(int argc, char* argv[])
{
    cli::Parser parser(argc, argv);
    parser.set_optional<size_t>("size", "size", DEFAULT_BYTES, "number of input bytes");
    parser.set_optional<int>("trials", "trials", -1, "number of iterations");
    parser.set_optional<std::string>("name_format",
                                     "name_format",
                                     "human",
                                     "either: json,human,txt");
    parser.set_optional<std::string>("seed", "seed", "random", get_seed_message());
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
    bench_naming::set_format(parser.get<std::string>("name_format"));
    const std::string  seed_type = parser.get<std::string>("seed");
    const managed_seed seed(seed_type);

    // HIP
    hipStream_t stream = 0; // default

    // Benchmark info
    add_common_benchmark_info();
    benchmark::AddCustomContext("size", std::to_string(size));
    benchmark::AddCustomContext("seed", seed_type);

    // Add benchmarks
    std::vector<benchmark::internal::Benchmark*> benchmarks{};
#ifdef BENCHMARK_CONFIG_TUNING
    const int parallel_instance  = parser.get<int>("parallel_instance");
    const int parallel_instances = parser.get<int>("parallel_instances");
    config_autotune_register::register_benchmark_subset(benchmarks,
                                                        parallel_instance,
                                                        parallel_instances,
                                                        size,
                                                        seed,
                                                        stream);
#else // BENCHMARK_CONFIG_TUNING \
    // add_adjacent_find_benchmarks(benchmarks, size, seed, stream);
    using custom_float2          = custom_type<float, float>;
    using custom_double2         = custom_type<double, double>;
    using custom_int2            = custom_type<int, int>;
    using custom_char_double     = custom_type<char, double>;
    using custom_longlong_double = custom_type<long long, double>;

    // Tuned types
    CREATE_ADJACENT_FIND_BENCHMARKS(int8_t)
    CREATE_ADJACENT_FIND_BENCHMARKS(int16_t)
    CREATE_ADJACENT_FIND_BENCHMARKS(int32_t)
    CREATE_ADJACENT_FIND_BENCHMARKS(int64_t)
    CREATE_ADJACENT_FIND_BENCHMARKS(rocprim::half)
    CREATE_ADJACENT_FIND_BENCHMARKS(float)
    CREATE_ADJACENT_FIND_BENCHMARKS(double)
    // Custom types
    CREATE_ADJACENT_FIND_BENCHMARKS(custom_float2)
    CREATE_ADJACENT_FIND_BENCHMARKS(custom_double2)
    CREATE_ADJACENT_FIND_BENCHMARKS(custom_int2)
    CREATE_ADJACENT_FIND_BENCHMARKS(custom_char_double)
    CREATE_ADJACENT_FIND_BENCHMARKS(custom_longlong_double)
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
