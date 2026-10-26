// MIT License
//
// Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "benchmark_device_search_n.hpp"
#include "primbench.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <string>
#include <vector>

#define CREATE_BENCHMARK(T, S, C) executor.queue<device_search_n_benchmark<T, S, C>>();

#define CREATE_BENCHMARKS(T)                                  \
    CREATE_BENCHMARK(T, size_t, count_equal_to<1>)            \
    CREATE_BENCHMARK(T, size_t, count_equal_to<6>)            \
    CREATE_BENCHMARK(T, size_t, count_equal_to<10>)           \
    CREATE_BENCHMARK(T, size_t, count_equal_to<14>)           \
    CREATE_BENCHMARK(T, size_t, count_equal_to<25>)           \
    CREATE_BENCHMARK(T, size_t, count_is_percent_of_size<50>) \
    CREATE_BENCHMARK(T, size_t, count_is_percent_of_size<100>)

int main(int argc, char* argv[])
{
    primbench::settings settings;
    settings.size = 2 * primbench::GiB;
    primbench::executor executor(argc, argv, settings, primbench::flags::sync);

#ifndef BENCHMARK_CONFIG_TUNING
    // Tuned types
    CREATE_BENCHMARKS(rocprim::int128_t)
    CREATE_BENCHMARKS(int64_t)
    CREATE_BENCHMARKS(int32_t)
    CREATE_BENCHMARKS(int16_t)
    CREATE_BENCHMARKS(int8_t)
    CREATE_BENCHMARKS(double)
    CREATE_BENCHMARKS(float)
    CREATE_BENCHMARKS(rocprim::half)

    #ifndef BENCHMARK_AUTOTUNED_TYPES_ONLY
    // Not tuned types
    CREATE_BENCHMARKS(rocprim::uint128_t)

    // Not tuned custom types
    CREATE_BENCHMARKS(custom_i32_i32)
    CREATE_BENCHMARKS(custom_i64_f64)
    #endif
#endif

    // Run benchmarks
    executor.run();
}
