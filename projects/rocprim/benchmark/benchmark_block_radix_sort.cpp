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

#include "benchmark_block_radix_sort.hpp"

#include "primbench.hpp"

#define CREATE_BENCHMARK(T, BS, RB, IPT) \
    executor.queue<block_radix_sort_benchmark<T, BenchmarkKind, BS, RB, IPT>>();

#define BENCHMARK_TYPE(type, block, radix_bits)  \
    CREATE_BENCHMARK(type, block, radix_bits, 1) \
    CREATE_BENCHMARK(type, block, radix_bits, 2) \
    CREATE_BENCHMARK(type, block, radix_bits, 3) \
    CREATE_BENCHMARK(type, block, radix_bits, 4) \
    CREATE_BENCHMARK(type, block, radix_bits, 8)

#define QUEUE_BENCHMARK(block, radix_bits)                                             \
    benchmark_types::queue_type<(benchmark_types::Type_Category::type_int32            \
                                 | benchmark_types::Type_Category::integer_8           \
                                 | benchmark_types::Type_Category::type_half           \
                                 | benchmark_types::Type_Category::type_custom_i32_i32 \
                                 | benchmark_types::Type_Category::type_int64          \
                                 | benchmark_types::Type_Category::integer_128)>(      \
        executor,                                                                      \
        [&](auto type_tag)                                                             \
        { BENCHMARK_TYPE(typename decltype(type_tag)::type, block, radix_bits) });

template<benchmark_kinds BenchmarkKind>
void add_benchmarks(primbench::executor& executor)
{
    QUEUE_BENCHMARK(64, 3)
    QUEUE_BENCHMARK(512, 3)

    QUEUE_BENCHMARK(64, 4)
    QUEUE_BENCHMARK(128, 4)
    QUEUE_BENCHMARK(192, 4)
    QUEUE_BENCHMARK(256, 4)
    QUEUE_BENCHMARK(320, 4)
    QUEUE_BENCHMARK(512, 4)
}

int main(int argc, char* argv[])
{
    primbench::settings settings;
    settings.size = 512 * primbench::MiB;
    primbench::executor executor(argc, argv, settings);

    add_benchmarks<benchmark_kinds::sort_keys>(executor);
    add_benchmarks<benchmark_kinds::sort_pairs>(executor);

    executor.run();
}
