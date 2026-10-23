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

#include "benchmark_block_reduce.hpp"

#include "primbench.hpp"

#define CREATE_BENCHMARK(T, BS, IPT) \
    executor.queue<block_reduce_benchmark<Benchmark, T, BS, IPT>>();

#define BENCHMARK_TYPE(type, block)   \
    CREATE_BENCHMARK(type, block, 1)  \
    CREATE_BENCHMARK(type, block, 2)  \
    CREATE_BENCHMARK(type, block, 3)  \
    CREATE_BENCHMARK(type, block, 4)  \
    CREATE_BENCHMARK(type, block, 8)  \
    CREATE_BENCHMARK(type, block, 11) \
    CREATE_BENCHMARK(type, block, 16)

template<typename Benchmark>
void add_benchmarks(primbench::executor& executor)
{
    // When block size is less than or equal to warp size
    BENCHMARK_TYPE(int32_t, 64)
    BENCHMARK_TYPE(float, 64)
    BENCHMARK_TYPE(double, 64)
    BENCHMARK_TYPE(int8_t, 64)
    BENCHMARK_TYPE(uint8_t, 64)
    BENCHMARK_TYPE(rocprim::half, 64)
    BENCHMARK_TYPE(rocprim::int128_t, 64)
    BENCHMARK_TYPE(rocprim::uint128_t, 64)

    BENCHMARK_TYPE(int32_t, 256)
    BENCHMARK_TYPE(float, 256)
    BENCHMARK_TYPE(double, 256)
    BENCHMARK_TYPE(int8_t, 256)
    BENCHMARK_TYPE(uint8_t, 256)
    BENCHMARK_TYPE(rocprim::half, 256)
    BENCHMARK_TYPE(rocprim::int128_t, 256)
    BENCHMARK_TYPE(rocprim::uint128_t, 256)

    CREATE_BENCHMARK(custom_f32_f32, 256, 1)
    CREATE_BENCHMARK(custom_f32_f32, 256, 4)
    CREATE_BENCHMARK(custom_f32_f32, 256, 8)

    CREATE_BENCHMARK(float2, 256, 1)
    CREATE_BENCHMARK(float2, 256, 4)
    CREATE_BENCHMARK(float2, 256, 8)

    CREATE_BENCHMARK(custom_f64_f64, 256, 1)
    CREATE_BENCHMARK(custom_f64_f64, 256, 4)
    CREATE_BENCHMARK(custom_f64_f64, 256, 8)

    CREATE_BENCHMARK(double2, 256, 1)
    CREATE_BENCHMARK(double2, 256, 4)
    CREATE_BENCHMARK(double2, 256, 8)

    CREATE_BENCHMARK(float4, 256, 1)
    CREATE_BENCHMARK(float4, 256, 4)
    CREATE_BENCHMARK(float4, 256, 8)
}

int main(int argc, char* argv[])
{
    primbench::settings settings;
    settings.size = 128 * primbench::MiB;
    primbench::executor executor(argc, argv, settings);

    using reduce_uwr_t = reduce<rocprim::block_reduce_algorithm::using_warp_reduce>;
    add_benchmarks<reduce_uwr_t>(executor);

    using reduce_rr_t = reduce<rocprim::block_reduce_algorithm::raking_reduce>;
    add_benchmarks<reduce_rr_t>(executor);

    using reduce_rrco_t = reduce<rocprim::block_reduce_algorithm::raking_reduce_commutative_only>;
    add_benchmarks<reduce_rrco_t>(executor);

    executor.run();
}
