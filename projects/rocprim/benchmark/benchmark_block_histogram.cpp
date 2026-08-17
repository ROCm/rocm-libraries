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

#include "benchmark_block_histogram.hpp"

#include "primbench.hpp"

#define CREATE_BENCHMARK(Benchmark, T, BS, IPT) \
    executor.queue<block_histogram_benchmark<Benchmark, T, BS, IPT>>();

#define BENCHMARK_TYPE(Benchmark, T, BS)  \
    CREATE_BENCHMARK(Benchmark, T, BS, 1) \
    CREATE_BENCHMARK(Benchmark, T, BS, 2) \
    CREATE_BENCHMARK(Benchmark, T, BS, 3) \
    CREATE_BENCHMARK(Benchmark, T, BS, 4) \
    CREATE_BENCHMARK(Benchmark, T, BS, 8) \
    CREATE_BENCHMARK(Benchmark, T, BS, 16)

#define BENCHMARK_TYPE_128(Benchmark, T, BS) \
    CREATE_BENCHMARK(Benchmark, T, BS, 1)    \
    CREATE_BENCHMARK(Benchmark, T, BS, 2)    \
    CREATE_BENCHMARK(Benchmark, T, BS, 3)    \
    CREATE_BENCHMARK(Benchmark, T, BS, 4)    \
    CREATE_BENCHMARK(Benchmark, T, BS, 8)    \
    CREATE_BENCHMARK(Benchmark, T, BS, 12)

#define BENCHMARK_ATOMIC()                            \
    BENCHMARK_TYPE(histogram_atomic_t, int32_t, 256)  \
    BENCHMARK_TYPE(histogram_atomic_t, int32_t, 320)  \
    BENCHMARK_TYPE(histogram_atomic_t, int32_t, 512)  \
                                                      \
    BENCHMARK_TYPE(histogram_atomic_t, uint64_t, 256) \
    BENCHMARK_TYPE(histogram_atomic_t, uint64_t, 320)

#define BENCHMARK_SORT()                                         \
    BENCHMARK_TYPE(histogram_sort_t, int32_t, 256)               \
    BENCHMARK_TYPE(histogram_sort_t, int32_t, 320)               \
    BENCHMARK_TYPE(histogram_sort_t, int32_t, 512)               \
                                                                 \
    BENCHMARK_TYPE(histogram_sort_t, uint64_t, 256)              \
    BENCHMARK_TYPE(histogram_sort_t, uint64_t, 320)              \
                                                                 \
    BENCHMARK_TYPE_128(histogram_sort_t, rocprim::int128_t, 256) \
    BENCHMARK_TYPE_128(histogram_sort_t, rocprim::uint128_t, 256)

int main(int argc, char* argv[])
{
    primbench::settings settings;
    settings.size = 512 * primbench::MiB;
    primbench::executor executor(argc, argv, settings);

#ifndef BENCHMARK_CONFIG_TUNING
    using histogram_atomic_t = histogram<rocprim::block_histogram_algorithm::using_atomic>;
    using histogram_sort_t   = histogram<rocprim::block_histogram_algorithm::using_sort>;

    BENCHMARK_ATOMIC()
    BENCHMARK_SORT()
#endif

    executor.run();
}
