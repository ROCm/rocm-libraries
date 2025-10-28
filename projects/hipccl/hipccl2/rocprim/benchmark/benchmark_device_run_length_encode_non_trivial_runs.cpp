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

#include "benchmark_device_run_length_encode_non_trivial_runs.parallel.hpp"
#include "benchmark_utils.hpp"

// CmdParser
#include "cmdparser.hpp"

#include "../common/utils_custom_type.hpp"

// Google Benchmark
#include <benchmark/benchmark.h>

// HIP API
#include <hip/hip_runtime.h>

// rocPRIM
#include <rocprim/types.hpp>

#include <cstddef>
#include <stdint.h>
#include <string>
#include <vector>

#define CREATE_NON_TRIVIAL_RUNS_BENCHMARK(T)                              \
    executor.queue_instance(device_non_trivial_runs_benchmark<T, 16>());  \
    executor.queue_instance(device_non_trivial_runs_benchmark<T, 256>()); \
    executor.queue_instance(device_non_trivial_runs_benchmark<T, 4096>());

int main(int argc, char* argv[])
{
    benchmark_utils::executor executor(argc, argv, 2 * benchmark_utils::GiB, 10, 5);

#ifndef BENCHMARK_CONFIG_TUNING
    // Tuned types
    CREATE_NON_TRIVIAL_RUNS_BENCHMARK(rocprim::int128_t)
    CREATE_NON_TRIVIAL_RUNS_BENCHMARK(int64_t)
    CREATE_NON_TRIVIAL_RUNS_BENCHMARK(int32_t)
    CREATE_NON_TRIVIAL_RUNS_BENCHMARK(int16_t)
    CREATE_NON_TRIVIAL_RUNS_BENCHMARK(int8_t)
    CREATE_NON_TRIVIAL_RUNS_BENCHMARK(double)
    CREATE_NON_TRIVIAL_RUNS_BENCHMARK(float)
    CREATE_NON_TRIVIAL_RUNS_BENCHMARK(rocprim::half)

    #ifndef BENCHMARK_AUTOTUNED_TYPES_ONLY
    // Not tuned types
    CREATE_NON_TRIVIAL_RUNS_BENCHMARK(rocprim::uint128_t)

    // Not tuned custom types
    using custom_float2  = common::custom_type<float, float>;
    using custom_double2 = common::custom_type<double, double>;

    CREATE_NON_TRIVIAL_RUNS_BENCHMARK(custom_float2)
    CREATE_NON_TRIVIAL_RUNS_BENCHMARK(custom_double2)
    #endif
#endif

    executor.run();
}
