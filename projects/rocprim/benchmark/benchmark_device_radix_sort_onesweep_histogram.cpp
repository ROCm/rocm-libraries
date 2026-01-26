// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <hip/hip_runtime.h>

#include "benchmark_device_radix_sort_onesweep_histogram.hpp"
#include "primbench.hpp"

#include <cstddef>
#include <string>
#include <vector>

#define CREATE_BENCHMARK(...) \
    executor.queue<device_radix_sort_onesweep_histogram_benchmark<__VA_ARGS__>>();

#define CREATE_BENCHMARK_TYPE_TUNING(KeyType)      \
    CREATE_BENCHMARK(KeyType, rocprim::empty_type) \
    CREATE_BENCHMARK(KeyType, rocprim::int128_t)   \
    CREATE_BENCHMARK(KeyType, long long)           \
    CREATE_BENCHMARK(KeyType, int)                 \
    CREATE_BENCHMARK(KeyType, short)               \
    CREATE_BENCHMARK(KeyType, int8_t)

int main(int argc, char* argv[])
{
    primbench::executor executor(argc, argv, 128 * primbench::MiB);

#ifndef BENCHMARK_CONFIG_TUNING
    // Tuned types
    CREATE_BENCHMARK_TYPE_TUNING(rocprim::int128_t)
    CREATE_BENCHMARK_TYPE_TUNING(long long)
    CREATE_BENCHMARK_TYPE_TUNING(int)
    CREATE_BENCHMARK_TYPE_TUNING(short)
    CREATE_BENCHMARK_TYPE_TUNING(int8_t)
    CREATE_BENCHMARK_TYPE_TUNING(double)
    CREATE_BENCHMARK_TYPE_TUNING(float)
    CREATE_BENCHMARK_TYPE_TUNING(rocprim::half)

    #ifndef BENCHMARK_AUTOTUNED_TYPES_ONLY
    // Not tuned types
    CREATE_BENCHMARK(uint8_t)
    CREATE_BENCHMARK(rocprim::uint128_t)

    CREATE_BENCHMARK(int, float)
    CREATE_BENCHMARK(int, double)
    CREATE_BENCHMARK(int, float2)
    CREATE_BENCHMARK(int, double2)

    CREATE_BENCHMARK(long long, float)
    CREATE_BENCHMARK(long long, double)
    CREATE_BENCHMARK(long long, float2)
    CREATE_BENCHMARK(long long, double2)

    CREATE_BENCHMARK(uint8_t, uint8_t)
    CREATE_BENCHMARK(rocprim::half, rocprim::half)
    CREATE_BENCHMARK(rocprim::uint128_t, rocprim::uint128_t)

    // Not tuned custom types
    CREATE_BENCHMARK(int, custom_float2)
    CREATE_BENCHMARK(int, custom_double2)
    CREATE_BENCHMARK(long long, custom_float2)
    CREATE_BENCHMARK(long long, custom_double2)
    #endif
#endif

    executor.run();
}
