// MIT License
//
// Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "benchmark_device_histogram.hpp"

#include "primbench.hpp"

#define CREATE_EVEN_BENCHMARK(T, BINS, SCALE) \
    executor.queue<device_histogram_even_benchmark<T>>(BINS, SCALE, entropy_reduction);

#define BENCHMARK_EVEN_TYPE(T, S)     \
    CREATE_EVEN_BENCHMARK(T, 10, S)   \
    CREATE_EVEN_BENCHMARK(T, 100, S)  \
    CREATE_EVEN_BENCHMARK(T, 1000, S) \
    CREATE_EVEN_BENCHMARK(T, 10000, S)

#define CREATE_MULTI_EVEN_BENCHMARK(CHANNELS, ACTIVE_CHANNELS, T, BINS, SCALE)           \
    executor.queue<device_multi_histogram_even_benchmark<T, CHANNELS, ACTIVE_CHANNELS>>( \
        BINS,                                                                            \
        SCALE,                                                                           \
        entropy_reduction);

#define BENCHMARK_MULTI_EVEN_TYPE(C, A, T, S)     \
    CREATE_MULTI_EVEN_BENCHMARK(C, A, T, 10, S)   \
    CREATE_MULTI_EVEN_BENCHMARK(C, A, T, 100, S)  \
    CREATE_MULTI_EVEN_BENCHMARK(C, A, T, 1000, S) \
    CREATE_MULTI_EVEN_BENCHMARK(C, A, T, 10000, S)

#define CREATE_RANGE_BENCHMARK(T, BINS) executor.queue<device_histogram_range_benchmark<T>>(BINS);

#define BENCHMARK_RANGE_TYPE(T)     \
    CREATE_RANGE_BENCHMARK(T, 10)   \
    CREATE_RANGE_BENCHMARK(T, 100)  \
    CREATE_RANGE_BENCHMARK(T, 1000) \
    CREATE_RANGE_BENCHMARK(T, 10000)

#define CREATE_MULTI_RANGE_BENCHMARK(CHANNELS, ACTIVE_CHANNELS, T, BINS) \
    executor.queue<device_multi_histogram_range_benchmark<T, CHANNELS, ACTIVE_CHANNELS>>(BINS);

#define BENCHMARK_MULTI_RANGE_TYPE(C, A, T)     \
    CREATE_MULTI_RANGE_BENCHMARK(C, A, T, 10)   \
    CREATE_MULTI_RANGE_BENCHMARK(C, A, T, 100)  \
    CREATE_MULTI_RANGE_BENCHMARK(C, A, T, 1000) \
    CREATE_MULTI_RANGE_BENCHMARK(C, A, T, 10000)

int main(int argc, char* argv[])
{
    primbench::settings settings;
    settings.size = 128 * primbench::MiB;
    primbench::executor executor(argc, argv, settings);

#ifndef BENCHMARK_CONFIG_TUNING
    const int entropy_reductions[] = {0, 2, 4, 6};

    // Even benchmarks
    for(int entropy_reduction : entropy_reductions)
    {
        BENCHMARK_EVEN_TYPE(int64_t, 12345)
        BENCHMARK_EVEN_TYPE(int32_t, 1234)
        BENCHMARK_EVEN_TYPE(int16_t, 5)
        CREATE_EVEN_BENCHMARK(uint8_t, 16, 16)
        CREATE_EVEN_BENCHMARK(uint8_t, 256, 1)
        BENCHMARK_EVEN_TYPE(double, 1234)
        BENCHMARK_EVEN_TYPE(float, 1234)
        BENCHMARK_EVEN_TYPE(rocprim::half, 5)
        CREATE_EVEN_BENCHMARK(rocprim::int128_t, 16, 16)
        CREATE_EVEN_BENCHMARK(rocprim::int128_t, 256, 1)
        CREATE_EVEN_BENCHMARK(rocprim::uint128_t, 16, 16)
        CREATE_EVEN_BENCHMARK(rocprim::uint128_t, 256, 1)
    }

    // Multi-even benchmarks
    for(int entropy_reduction : entropy_reductions)
    {
        BENCHMARK_MULTI_EVEN_TYPE(4, 4, int32_t, 1234)
        BENCHMARK_MULTI_EVEN_TYPE(4, 3, int16_t, 5)
        CREATE_MULTI_EVEN_BENCHMARK(4, 3, uint8_t, 16, 16)
        CREATE_MULTI_EVEN_BENCHMARK(4, 3, uint8_t, 256, 1)
        BENCHMARK_MULTI_EVEN_TYPE(3, 3, float, 1234)
        CREATE_MULTI_EVEN_BENCHMARK(4, 3, rocprim::int128_t, 16, 16)
        CREATE_MULTI_EVEN_BENCHMARK(4, 3, rocprim::int128_t, 256, 1)
        CREATE_MULTI_EVEN_BENCHMARK(4, 3, rocprim::uint128_t, 16, 16)
        CREATE_MULTI_EVEN_BENCHMARK(4, 3, rocprim::uint128_t, 256, 1)
    }

    // Range benchmarks
    BENCHMARK_RANGE_TYPE(int64_t)
    BENCHMARK_RANGE_TYPE(int32_t)
    BENCHMARK_RANGE_TYPE(int16_t)
    CREATE_RANGE_BENCHMARK(uint8_t, 16)
    CREATE_RANGE_BENCHMARK(uint8_t, 256)
    BENCHMARK_RANGE_TYPE(double)
    BENCHMARK_RANGE_TYPE(float)
    BENCHMARK_RANGE_TYPE(rocprim::half)
    CREATE_RANGE_BENCHMARK(rocprim::int128_t, 16)
    CREATE_RANGE_BENCHMARK(rocprim::int128_t, 256)
    CREATE_RANGE_BENCHMARK(rocprim::uint128_t, 16)
    CREATE_RANGE_BENCHMARK(rocprim::uint128_t, 256)

    // Multi-range benchmarks
    BENCHMARK_MULTI_RANGE_TYPE(4, 4, int32_t)
    BENCHMARK_MULTI_RANGE_TYPE(4, 3, int16_t)
    CREATE_MULTI_RANGE_BENCHMARK(4, 3, uint8_t, 16)
    CREATE_MULTI_RANGE_BENCHMARK(4, 3, uint8_t, 256)
    BENCHMARK_MULTI_RANGE_TYPE(3, 3, float)
    BENCHMARK_MULTI_RANGE_TYPE(2, 2, double)
    CREATE_MULTI_RANGE_BENCHMARK(4, 3, rocprim::int128_t, 16)
    CREATE_MULTI_RANGE_BENCHMARK(4, 3, rocprim::int128_t, 256)
    CREATE_MULTI_RANGE_BENCHMARK(4, 3, rocprim::uint128_t, 16)
    CREATE_MULTI_RANGE_BENCHMARK(4, 3, rocprim::uint128_t, 256)
#endif

    executor.run();
}
