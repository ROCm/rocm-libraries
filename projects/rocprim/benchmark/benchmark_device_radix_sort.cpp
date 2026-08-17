// MIT License
//
// Copyright (c) 2017-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "benchmark_device_radix_sort.hpp"
#include "primbench.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <string>
#include <vector>

#define CREATE_RADIX_SORT_BENCHMARK(...) executor.queue<device_radix_sort_benchmark<__VA_ARGS__>>();

int main(int argc, char* argv[])
{
    primbench::settings settings;
    settings.size = 128 * primbench::MiB;
    primbench::executor executor(argc, argv, settings);

    CREATE_RADIX_SORT_BENCHMARK(int32_t)
    CREATE_RADIX_SORT_BENCHMARK(float)
    CREATE_RADIX_SORT_BENCHMARK(int64_t)
    CREATE_RADIX_SORT_BENCHMARK(int8_t)
    CREATE_RADIX_SORT_BENCHMARK(uint8_t)
    CREATE_RADIX_SORT_BENCHMARK(rocprim::half)
    CREATE_RADIX_SORT_BENCHMARK(int16_t)
    CREATE_RADIX_SORT_BENCHMARK(custom_f32_i16)
    CREATE_RADIX_SORT_BENCHMARK(rocprim::int128_t)
    CREATE_RADIX_SORT_BENCHMARK(rocprim::uint128_t)

    CREATE_RADIX_SORT_BENCHMARK(int32_t, float)
    CREATE_RADIX_SORT_BENCHMARK(int32_t, double)
    CREATE_RADIX_SORT_BENCHMARK(int32_t, float2)
    CREATE_RADIX_SORT_BENCHMARK(int32_t, custom_f32_f32)
    CREATE_RADIX_SORT_BENCHMARK(int32_t, double2)
    CREATE_RADIX_SORT_BENCHMARK(int32_t, custom_f64_f64)

    CREATE_RADIX_SORT_BENCHMARK(int64_t, float)
    CREATE_RADIX_SORT_BENCHMARK(int64_t, double)
    CREATE_RADIX_SORT_BENCHMARK(int64_t, float2)
    CREATE_RADIX_SORT_BENCHMARK(int64_t, custom_f32_f32)
    CREATE_RADIX_SORT_BENCHMARK(int64_t, double2)
    CREATE_RADIX_SORT_BENCHMARK(int64_t, custom_f64_f64)
    CREATE_RADIX_SORT_BENCHMARK(int8_t, int8_t)
    CREATE_RADIX_SORT_BENCHMARK(uint8_t, uint8_t)
    CREATE_RADIX_SORT_BENCHMARK(rocprim::half, rocprim::half)
    CREATE_RADIX_SORT_BENCHMARK(custom_f32_i16, double)
    CREATE_RADIX_SORT_BENCHMARK(rocprim::int128_t, rocprim::int128_t)
    CREATE_RADIX_SORT_BENCHMARK(rocprim::uint128_t, rocprim::uint128_t)

    executor.run();
}
