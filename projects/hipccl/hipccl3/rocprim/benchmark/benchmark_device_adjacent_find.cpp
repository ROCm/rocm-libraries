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

#include "benchmark_device_adjacent_find.parallel.hpp"
#include "benchmark_utils.hpp"

#ifndef BENCHMARK_CONFIG_TUNING
    #include "../common/utils_custom_type.hpp"
#endif

// HIP
#include <hip/hip_runtime.h>

#ifndef BENCHMARK_CONFIG_TUNING
    #include <rocprim/types.hpp>
#endif

// C++ Standard Library
#include <cstddef>
#include <string>
#include <vector>
#ifndef BENCHMARK_CONFIG_TUNING
    #include <stdint.h>
#endif

#define CREATE_BENCHMARK(T, P) executor.queue_instance(device_adjacent_find_benchmark<T, P>());

#define CREATE_ADJACENT_FIND_BENCHMARKS(T) \
    CREATE_BENCHMARK(T, 1)                 \
    CREATE_BENCHMARK(T, 5)                 \
    CREATE_BENCHMARK(T, 9)

int main(int argc, char* argv[])
{
    benchmark_utils::executor executor(argc, argv, 2 * benchmark_utils::GiB, 10, 5);

#ifndef BENCHMARK_CONFIG_TUNING
    using custom_float2          = common::custom_type<float, float>;
    using custom_double2         = common::custom_type<double, double>;
    using custom_int2            = common::custom_type<int, int>;
    using custom_char_double     = common::custom_type<char, double>;
    using custom_longlong_double = common::custom_type<long long, double>;

    // Tuned types
    CREATE_ADJACENT_FIND_BENCHMARKS(int8_t)
    CREATE_ADJACENT_FIND_BENCHMARKS(int16_t)
    CREATE_ADJACENT_FIND_BENCHMARKS(int32_t)
    CREATE_ADJACENT_FIND_BENCHMARKS(int64_t)
    CREATE_ADJACENT_FIND_BENCHMARKS(rocprim::half)
    CREATE_ADJACENT_FIND_BENCHMARKS(float)
    CREATE_ADJACENT_FIND_BENCHMARKS(double)
    CREATE_ADJACENT_FIND_BENCHMARKS(rocprim::int128_t)
    CREATE_ADJACENT_FIND_BENCHMARKS(rocprim::uint128_t)

    // Custom types
    CREATE_ADJACENT_FIND_BENCHMARKS(custom_float2)
    CREATE_ADJACENT_FIND_BENCHMARKS(custom_double2)
    CREATE_ADJACENT_FIND_BENCHMARKS(custom_int2)
    CREATE_ADJACENT_FIND_BENCHMARKS(custom_char_double)
    CREATE_ADJACENT_FIND_BENCHMARKS(custom_longlong_double)
#endif

    executor.run();
}
