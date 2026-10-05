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

#include "benchmark_block_exchange.hpp"

#include "benchmark_utils.hpp"
#include "primbench.hpp"

#define QUEUE_BENCHMARK(BS, IPT)                                                                \
    benchmark_types::queue_type<((benchmark_types::Type_Category::integer_signed                \
                                  ^ benchmark_types::Type_Category::type_int16)                 \
                                 | benchmark_types::Type_Category::integer_128                  \
                                 | benchmark_types::Type_Category::type_half                    \
                                 | benchmark_types::Type_Category::custom_floating_point        \
                                 | benchmark_types::Type_Category::rocprim_vector)>(            \
        executor,                                                                               \
        [&](auto type_tag)                                                                      \
        {                                                                                       \
            executor.template queue<block_exchange_benchmark<Benchmark,                         \
                                                             typename decltype(type_tag)::type, \
                                                             BS,                                \
                                                             IPT>>();                           \
        });
template<typename Benchmark>
void add_benchmarks(primbench::executor& executor)
{
    QUEUE_BENCHMARK(256, 1)
    QUEUE_BENCHMARK(256, 2)
    QUEUE_BENCHMARK(256, 3)
    QUEUE_BENCHMARK(256, 4)
    QUEUE_BENCHMARK(256, 7)
    QUEUE_BENCHMARK(256, 8)
}

int main(int argc, char* argv[])
{
    primbench::settings settings;
    settings.size = 128 * primbench::MiB;
    primbench::executor executor(argc, argv, settings);

    add_benchmarks<blocked_to_striped>(executor);
    add_benchmarks<striped_to_blocked>(executor);
    add_benchmarks<blocked_to_warp_striped>(executor);
    add_benchmarks<warp_striped_to_blocked>(executor);
    add_benchmarks<scatter_to_blocked>(executor);
    add_benchmarks<scatter_to_striped>(executor);

    executor.run();
}
