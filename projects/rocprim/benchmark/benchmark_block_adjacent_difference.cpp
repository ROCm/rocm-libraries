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

constexpr auto crosslane
    = rocprim::block_adjacent_difference_algorithm::adjacent_difference_crosslane;
constexpr auto shared_mem
    = rocprim::block_adjacent_difference_algorithm::adjacent_difference_shared_mem;

#define CREATE_BENCHMARK(T, BS, IPT, WITH_TILE, ALGO) \
    executor.queue<block_adjacent_difference_benchmark<Benchmark, T, BS, IPT, WITH_TILE, ALGO>>();

#define CREATE_BENCHMARK_KINDS(T, BS, IPT, WITH_TILE)  \
    CREATE_BENCHMARK(T, BS, IPT, WITH_TILE, crosslane) \
    CREATE_BENCHMARK(T, BS, IPT, WITH_TILE, shared_mem)

#define BENCHMARK_TYPE(T, BS, WITH_TILE)         \
    CREATE_BENCHMARK_KINDS(T, BS, 1, WITH_TILE)  \
    CREATE_BENCHMARK_KINDS(T, BS, 3, WITH_TILE)  \
    CREATE_BENCHMARK_KINDS(T, BS, 4, WITH_TILE)  \
    CREATE_BENCHMARK_KINDS(T, BS, 8, WITH_TILE)  \
    CREATE_BENCHMARK_KINDS(T, BS, 16, WITH_TILE) \
    CREATE_BENCHMARK_KINDS(T, BS, 32, WITH_TILE)

template<typename Benchmark>
void add_benchmarks(primbench::executor& executor)
{
    BENCHMARK_TYPE(int32_t, 256, false)
    BENCHMARK_TYPE(float, 256, false)
    BENCHMARK_TYPE(int8_t, 256, false)
    BENCHMARK_TYPE(rocprim::half, 256, false)
    BENCHMARK_TYPE(int64_t, 256, false)
    BENCHMARK_TYPE(double, 256, false)
    BENCHMARK_TYPE(rocprim::int128_t, 256, false)
    BENCHMARK_TYPE(rocprim::uint128_t, 256, false)

    if(!std::is_same<Benchmark, subtract_right_partial>::value)
    {
        BENCHMARK_TYPE(int32_t, 256, true)
        BENCHMARK_TYPE(float, 256, true)
        BENCHMARK_TYPE(int8_t, 256, true)
        BENCHMARK_TYPE(rocprim::half, 256, true)
        BENCHMARK_TYPE(int64_t, 256, true)
        BENCHMARK_TYPE(double, 256, true)
        BENCHMARK_TYPE(rocprim::int128_t, 256, true)
        BENCHMARK_TYPE(rocprim::uint128_t, 256, true)
    }
}

int main(int argc, char* argv[])
{
    primbench::settings settings;
    settings.size                    = 512 * primbench::MiB;
    settings.min_gpu_ms_per_batch    = 100;
    settings.noise_tolerance_percent = 2;
    primbench::executor executor(argc, argv, settings);

    add_benchmarks<subtract_left>(executor);
    add_benchmarks<subtract_right>(executor);
    add_benchmarks<subtract_left_partial>(executor);
    add_benchmarks<subtract_right_partial>(executor);

    executor.run();
}
