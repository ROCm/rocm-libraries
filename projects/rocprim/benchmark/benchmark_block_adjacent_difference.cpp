// MIT License
//
// Copyright (c) 2022-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "benchmark_block_adjacent_difference.hpp"

#include "primbench.hpp"

#define CREATE_BENCHMARK(T, BS, IPT, WITH_TILE) \
    executor.queue<block_adjacent_difference_benchmark<Benchmark, T, BS, IPT, WITH_TILE>>();

#define BENCHMARK_TYPE(type, block, with_tile)              \
    CREATE_BENCHMARK(type, block, 1, with_tile, Algorithm)  \
    CREATE_BENCHMARK(type, block, 3, with_tile, Algorithm)  \
    CREATE_BENCHMARK(type, block, 4, with_tile, Algorithm)  \
    CREATE_BENCHMARK(type, block, 8, with_tile, Algorithm)  \
    CREATE_BENCHMARK(type, block, 16, with_tile, Algorithm) \
    CREATE_BENCHMARK(type, block, 32, with_tile, Algorithm)

template<typename Benchmark>
void add_benchmarks(primbench::executor& executor)
{
    BENCHMARK_TYPE(int, 256, false)
    BENCHMARK_TYPE(float, 256, false)
    BENCHMARK_TYPE(int8_t, 256, false)
    BENCHMARK_TYPE(rocprim::half, 256, false)
    BENCHMARK_TYPE(long long, 256, false)
    BENCHMARK_TYPE(double, 256, false)
    BENCHMARK_TYPE(rocprim::int128_t, 256, false)
    BENCHMARK_TYPE(rocprim::uint128_t, 256, false)

    if(Benchmark::enable_with_tile)
    {
        BENCHMARK_TYPE(int, 256, true)
        BENCHMARK_TYPE(float, 256, true)
        BENCHMARK_TYPE(int8_t, 256, true)
        BENCHMARK_TYPE(rocprim::half, 256, true)
        BENCHMARK_TYPE(long long, 256, true)
        BENCHMARK_TYPE(double, 256, true)
        BENCHMARK_TYPE(rocprim::int128_t, 256, true)
        BENCHMARK_TYPE(rocprim::uint128_t, 256, true)
    }
}

int main(int argc, char* argv[])
{
    primbench::executor executor(argc, argv, 512 * primbench::MiB);

    add_benchmarks<subtract_left>(executor);
    add_benchmarks<subtract_right>(executor);
    add_benchmarks<subtract_left_partial>(executor);
    add_benchmarks<subtract_right_partial>(executor);

    executor.run();
}
