// MIT License
//
// Copyright (c) 2017-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "benchmark_device_segmented_reduce.parallel.hpp"
#include "benchmark_utils.hpp"
// CmdParser
#include "cmdparser.hpp"

#include "../common/utils_custom_type.hpp"

// Google Benchmark
#include <benchmark/benchmark.h>

// HIP API
#include <hip/hip_runtime.h>

// rocPRIM
#include <rocprim/device/device_segmented_reduce.hpp>
#include <rocprim/functional.hpp>
#include <rocprim/types.hpp>

#include <cmath>
#include <cstddef>
#include <numeric>
#include <random>
#include <stdint.h>
#include <string>
#include <vector>

#ifndef DEFAULT_BYTES
const size_t DEFAULT_BYTES = 1024 * 1024 * 32 * 4;
#endif

#define ADD_BENCHMARK(T, SEGMENTS, INSTANCE)                                           \
    benchmark::internal::Benchmark* benchmark = benchmark::RegisterBenchmark(          \
        bench_naming::format_name("{lvl:device,algo:reduce_segmented,key_type:" #T     \
                                  ",segment_count:"                                    \
                                  + std::to_string(SEGMENTS) + ",cfg:default_config}") \
            .c_str(),                                                                  \
        [INSTANCE](benchmark::State&   state,                                          \
                   size_t              _desired_segments,                              \
                   size_t              _size,                                          \
                   const managed_seed& _seed,                                          \
                   hipStream_t         _stream)                                        \
        { INSTANCE.run_benchmark(state, _desired_segments, _size, _seed, _stream); },  \
        SEGMENTS,                                                                      \
        bytes,                                                                         \
        seed,                                                                          \
        stream);

#define CREATE_BENCHMARK(T, SEGMENTS)                        \
    {                                                        \
        const device_segmented_reduce_benchmark<T> instance; \
        ADD_BENCHMARK(T, SEGMENTS, instance)                 \
        benchmarks.emplace_back(benchmark);                  \
    }

#define BENCHMARK_TYPE(type)     \
    CREATE_BENCHMARK(type, 1)    \
    CREATE_BENCHMARK(type, 10)   \
    CREATE_BENCHMARK(type, 100)  \
    CREATE_BENCHMARK(type, 1000) \
    CREATE_BENCHMARK(type, 10000)

void add_benchmarks(std::vector<benchmark::internal::Benchmark*>& benchmarks,
                    size_t                                        bytes,
                    const managed_seed&                           seed,
                    hipStream_t                                   stream)
{
    using custom_float2  = common::custom_type<float, float>;
    using custom_double2 = common::custom_type<double, double>;

    BENCHMARK_TYPE(float)
    BENCHMARK_TYPE(double)
    BENCHMARK_TYPE(int8_t)
    BENCHMARK_TYPE(uint8_t)
    BENCHMARK_TYPE(rocprim::half)
    BENCHMARK_TYPE(int)
    BENCHMARK_TYPE(custom_float2)
    BENCHMARK_TYPE(custom_double2)
    BENCHMARK_TYPE(rocprim::int128_t)
    BENCHMARK_TYPE(rocprim::uint128_t)
}

int main(int argc, char* argv[])
{
    cli::Parser parser(argc, argv);
    parser.set_optional<size_t>("size", "size", DEFAULT_BYTES, "number of bytes");
    parser.set_optional<int>("trials", "trials", -1, "number of iterations");
    parser.set_optional<std::string>("name_format",
                                     "name_format",
                                     "human",
                                     "either: json,human,txt");
    // fixed seed as a random seed adds a lot of variance
    parser.set_optional<std::string>("seed", "seed", "321", get_seed_message());
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
    const size_t bytes  = parser.get<size_t>("size");
    const int    trials = parser.get<int>("trials");
    bench_naming::set_format(parser.get<std::string>("name_format"));
    const std::string  seed_type = parser.get<std::string>("seed");
    const managed_seed seed(seed_type);

    // HIP
    hipStream_t stream = 0; // default

    // Benchmark info
    add_common_benchmark_info();
    benchmark::AddCustomContext("bytes", std::to_string(bytes));
    benchmark::AddCustomContext("seed", seed_type);

    // Add benchmarks
    std::vector<benchmark::internal::Benchmark*> benchmarks = {};
#ifdef BENCHMARK_CONFIG_TUNING
    const int parallel_instance  = parser.get<int>("parallel_instance");
    const int parallel_instances = parser.get<int>("parallel_instances");
    config_autotune_register::register_benchmark_subset(benchmarks,
                                                        parallel_instance,
                                                        parallel_instances,
                                                        bytes,
                                                        seed,
                                                        stream);
#else
    add_benchmarks(benchmarks, bytes, seed, stream);
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
