// MIT License
//
// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "benchmark_device_search_n.parallel.hpp"
#include "benchmark_utils.hpp"

// HIP API
#include <hip/hip_runtime.h>

#include <cstddef>
#include <string>
#include <vector>

#define CREATE_BENCHMARK(T, S, C) executor.queue_instance(benchmark_search_n<T, S, C>());

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
    benchmark_utils::executor executor(argc, argv, 2 * benchmark_utils::GiB, 10, 10);

#ifndef BENCHMARK_CONFIG_TUNING
    using custom_int2            = common::custom_type<int>;
    using custom_longlong_double = common::custom_type<long long, double>;

    CREATE_BENCHMARKS(custom_int2)
    CREATE_BENCHMARKS(custom_longlong_double)
    CREATE_BENCHMARKS(int8_t)
    CREATE_BENCHMARKS(int16_t)
    CREATE_BENCHMARKS(int32_t)
    CREATE_BENCHMARKS(int64_t)
    CREATE_BENCHMARKS(rocprim::int128_t)
    CREATE_BENCHMARKS(rocprim::uint128_t)
    CREATE_BENCHMARKS(rocprim::half)
    CREATE_BENCHMARKS(float)
    CREATE_BENCHMARKS(double)
#endif

    // Run benchmarks
    executor.run();
}
