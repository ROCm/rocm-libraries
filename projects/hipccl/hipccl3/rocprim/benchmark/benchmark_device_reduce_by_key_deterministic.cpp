// MIT License
//
// Copyright (c) 2017-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "benchmark_device_reduce_by_key.parallel.hpp"
#include "benchmark_utils.hpp"
// CmdParser
#include "cmdparser.hpp"

// Google Benchmark
#include <benchmark/benchmark.h>

// HIP API
#include <hip/hip_runtime.h>

#include <cstddef>
#include <string>
#include <vector>

#ifndef DEFAULT_BYTES
constexpr size_t DEFAULT_BYTES = size_t{2} << 30; // 2 GiB
#endif

#define CREATE_BENCHMARK(KEY, VALUE, MAX_SEGMENT_LENGTH)                                     \
    {                                                                                        \
        const device_reduce_by_key_benchmark<KEY, VALUE, MAX_SEGMENT_LENGTH, true> instance; \
        REGISTER_BENCHMARK(benchmarks, size, seed, stream, instance);                        \
    }

#define CREATE_BENCHMARK_TYPE(KEY, VALUE) \
    CREATE_BENCHMARK(KEY, VALUE, 10);     \
    CREATE_BENCHMARK(KEY, VALUE, 1000)

// some of the tuned types
#define CREATE_BENCHMARK_TYPES(KEY)            \
    CREATE_BENCHMARK_TYPE(KEY, int8_t);        \
    CREATE_BENCHMARK_TYPE(KEY, rocprim::half); \
    CREATE_BENCHMARK_TYPE(KEY, int32_t);       \
    CREATE_BENCHMARK_TYPE(KEY, float);         \
    CREATE_BENCHMARK_TYPE(KEY, double)

// all of the tuned types
#define CREATE_BENCHMARK_TYPE_TUNING(KEY)      \
    CREATE_BENCHMARK_TYPE(KEY, int8_t);        \
    CREATE_BENCHMARK_TYPE(KEY, int16_t);       \
    CREATE_BENCHMARK_TYPE(KEY, int32_t);       \
    CREATE_BENCHMARK_TYPE(KEY, int64_t);       \
    CREATE_BENCHMARK_TYPE(KEY, rocprim::half); \
    CREATE_BENCHMARK_TYPE(KEY, float);         \
    CREATE_BENCHMARK_TYPE(KEY, double)

int main(int argc, char* argv[])
{
    cli::Parser parser(argc, argv);
    parser.set_optional<size_t>("size", "size", DEFAULT_BYTES, "number of bytes");
    parser.set_optional<int>("trials", "trials", -1, "number of iterations");
    parser.set_optional<std::string>("name_format",
                                     "name_format",
                                     "human",
                                     "either: json,human,txt");
    parser.set_optional<std::string>("seed", "seed", "random", get_seed_message());
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
    std::vector<benchmark::internal::Benchmark*> benchmarks = {};
    // tuned types
    CREATE_BENCHMARK_TYPES(int8_t);
    CREATE_BENCHMARK_TYPES(int16_t);
    CREATE_BENCHMARK_TYPE_TUNING(int32_t);
    CREATE_BENCHMARK_TYPE_TUNING(int64_t);
    CREATE_BENCHMARK_TYPES(rocprim::half);
    CREATE_BENCHMARK_TYPES(float);
    CREATE_BENCHMARK_TYPES(double);

    // custom types
    using custom_float2  = custom_type<float, float>;
    using custom_double2 = custom_type<double, double>;

    CREATE_BENCHMARK_TYPE(int, custom_float2);
    CREATE_BENCHMARK_TYPE(int, custom_double2);

    CREATE_BENCHMARK_TYPE(long long, custom_float2);
    CREATE_BENCHMARK_TYPE(long long, custom_double2);

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
