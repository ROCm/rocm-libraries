// MIT License
//
// Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
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

// HIP API
#include <hip/hip_runtime.h>

#include "benchmark_device_radix_sort_onesweep.parallel.hpp"
#include "benchmark_utils.hpp"

#include <cstddef>
#include <string>
#include <vector>

#define CREATE_RADIX_SORT_BENCHMARK(...) \
    executor.queue_instance(device_radix_sort_onesweep_benchmark<__VA_ARGS__>());

int main(int argc, char* argv[])
{
    benchmark_utils::executor executor(argc, argv, 128 * benchmark_utils::MiB, 10, 5);

#ifndef BENCHMARK_CONFIG_TUNING
    CREATE_RADIX_SORT_BENCHMARK(int)
    CREATE_RADIX_SORT_BENCHMARK(float)
    CREATE_RADIX_SORT_BENCHMARK(long long)
    CREATE_RADIX_SORT_BENCHMARK(int8_t)
    CREATE_RADIX_SORT_BENCHMARK(uint8_t)
    CREATE_RADIX_SORT_BENCHMARK(rocprim::half)
    CREATE_RADIX_SORT_BENCHMARK(short)
    CREATE_RADIX_SORT_BENCHMARK(rocprim::int128_t)
    CREATE_RADIX_SORT_BENCHMARK(rocprim::uint128_t)

    using custom_float2  = common::custom_type<float, float>;
    using custom_double2 = common::custom_type<double, double>;

    CREATE_RADIX_SORT_BENCHMARK(int, float)
    CREATE_RADIX_SORT_BENCHMARK(int, double)
    CREATE_RADIX_SORT_BENCHMARK(int, float2)
    CREATE_RADIX_SORT_BENCHMARK(int, custom_float2)
    CREATE_RADIX_SORT_BENCHMARK(int, double2)
    CREATE_RADIX_SORT_BENCHMARK(int, custom_double2)

    CREATE_RADIX_SORT_BENCHMARK(long long, float)
    CREATE_RADIX_SORT_BENCHMARK(long long, double)
    CREATE_RADIX_SORT_BENCHMARK(long long, float2)
    CREATE_RADIX_SORT_BENCHMARK(long long, custom_float2)
    CREATE_RADIX_SORT_BENCHMARK(long long, double2)
    CREATE_RADIX_SORT_BENCHMARK(long long, custom_double2)
    CREATE_RADIX_SORT_BENCHMARK(int8_t, int8_t)
    CREATE_RADIX_SORT_BENCHMARK(uint8_t, uint8_t)
    CREATE_RADIX_SORT_BENCHMARK(rocprim::half, rocprim::half)
    CREATE_RADIX_SORT_BENCHMARK(rocprim::int128_t, rocprim::int128_t)
    CREATE_RADIX_SORT_BENCHMARK(rocprim::uint128_t, rocprim::uint128_t)
#endif

    executor.run();
}
