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

#include "benchmark_block_discontinuity.hpp"

#include "benchmark_utils.hpp"
#include "primbench.hpp"

constexpr auto crosslane
    = rocprim::block_adjacent_difference_algorithm::adjacent_difference_crosslane;
constexpr auto shared_mem
    = rocprim::block_adjacent_difference_algorithm::adjacent_difference_shared_mem;

#define CREATE_BENCHMARK(T, BS, IPT, WITH_TILE, ALGO) \
    executor.queue<block_discontinuity_benchmark<Benchmark, T, BS, IPT, WITH_TILE, ALGO>>();

#define CREATE_BENCHMARK_KINDS(T, BS, IPT, WITH_TILE)  \
    CREATE_BENCHMARK(T, BS, IPT, WITH_TILE, crosslane) \
    CREATE_BENCHMARK(T, BS, IPT, WITH_TILE, shared_mem)

#define BENCHMARK_TYPE(T, BS, WITH_TILE)        \
    CREATE_BENCHMARK_KINDS(T, BS, 1, WITH_TILE) \
    CREATE_BENCHMARK_KINDS(T, BS, 2, WITH_TILE) \
    CREATE_BENCHMARK_KINDS(T, BS, 3, WITH_TILE) \
    CREATE_BENCHMARK_KINDS(T, BS, 4, WITH_TILE) \
    CREATE_BENCHMARK_KINDS(T, BS, 8, WITH_TILE)

#define QUEUE_BENCHMARK(BS, WITH_TILE)                                            \
    benchmark_types::queue_type<((benchmark_types::Type_Category::warp            \
                                  ^ benchmark_types::Type_Category::type_float32  \
                                  ^ benchmark_types::Type_Category::type_float64) \
                                 | benchmark_types::Type_Category::type_int64     \
                                 | benchmark_types::Type_Category::type_half)>(   \
        executor,                                                                 \
        [&](auto type_tag) { BENCHMARK_TYPE(typename decltype(type_tag)::type, BS, WITH_TILE) });

template<typename Benchmark>
void add_benchmarks(primbench::executor& executor)
{
    QUEUE_BENCHMARK(256, false)
    QUEUE_BENCHMARK(256, true)
}

int main(int argc, char* argv[])
{
    primbench::settings settings;
    settings.size = 512 * primbench::MiB;
    primbench::executor executor(argc, argv, settings);

    add_benchmarks<flag_heads>(executor);
    add_benchmarks<flag_tails>(executor);
    add_benchmarks<flag_heads_and_tails>(executor);

    executor.run();
}
