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

#include "benchmark_warp_scan.hpp"

#include "primbench.hpp"

#define CREATE_BENCHMARK(T, BS, WS) executor.queue<warp_scan_benchmark<T, BS, WS, Benchmark>>();

// clang-format off
#define BENCHMARK_TYPE(type)        \
    CREATE_BENCHMARK(type, 64, 64)  \
    CREATE_BENCHMARK(type, 128, 64) \
    CREATE_BENCHMARK(type, 256, 64) \
    CREATE_BENCHMARK(type, 256, 32) \
    CREATE_BENCHMARK(type, 256, 16) \
    CREATE_BENCHMARK(type, 63, 63)  \
    CREATE_BENCHMARK(type, 62, 31)  \
    CREATE_BENCHMARK(type, 60, 15)
// clang-format on

// clang-format off
#define BENCHMARK_TYPE_P2(type)     \
    CREATE_BENCHMARK(type, 64, 64)  \
    CREATE_BENCHMARK(type, 128, 64) \
    CREATE_BENCHMARK(type, 256, 64) \
    CREATE_BENCHMARK(type, 256, 32) \
    CREATE_BENCHMARK(type, 256, 16)
// clang-format on

template<typename Benchmark>
auto add_benchmarks(primbench::executor& executor)
    -> std::enable_if_t<std::is_same<Benchmark, inclusive_scan>::value
                        || std::is_same<Benchmark, exclusive_scan>::value>
{
    BENCHMARK_TYPE(int32_t)
    BENCHMARK_TYPE(float)
    BENCHMARK_TYPE(double)
    BENCHMARK_TYPE(int8_t)
    BENCHMARK_TYPE(uint8_t)
    BENCHMARK_TYPE(rocprim::half)
    BENCHMARK_TYPE(custom_f64_f64)
    BENCHMARK_TYPE(custom_i32_f64)
    BENCHMARK_TYPE(rocprim::int128_t)
    BENCHMARK_TYPE(rocprim::uint128_t)
}

template<typename Benchmark>
auto add_benchmarks(primbench::executor& executor)
    -> std::enable_if_t<std::is_same<Benchmark, broadcast>::value>
{
    BENCHMARK_TYPE_P2(int32_t)
    BENCHMARK_TYPE_P2(float)
    BENCHMARK_TYPE_P2(double)
    BENCHMARK_TYPE_P2(int8_t)
    BENCHMARK_TYPE_P2(uint8_t)
    BENCHMARK_TYPE_P2(rocprim::half)
    BENCHMARK_TYPE_P2(custom_f64_f64)
    BENCHMARK_TYPE_P2(custom_i32_f64)
    BENCHMARK_TYPE_P2(rocprim::int128_t)
    BENCHMARK_TYPE_P2(rocprim::uint128_t)
}

int main(int argc, char* argv[])
{
    primbench::settings settings;
    settings.size = 128 * primbench::MiB;
    primbench::executor executor(argc, argv, settings);

    add_benchmarks<inclusive_scan>(executor);
    add_benchmarks<exclusive_scan>(executor);
    add_benchmarks<broadcast>(executor);

    executor.run();
}
