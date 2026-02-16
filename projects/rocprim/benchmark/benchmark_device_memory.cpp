// MIT License
//
// Copyright (c) 2018-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "benchmark_device_memory.hpp"

#include "primbench.hpp"

#define CREATE_BENCHMARK(METHOD, OPERATION, T, BLOCK_SIZE, IPT) \
    executor.queue<device_memory_benchmark<T, BLOCK_SIZE, IPT, METHOD, OPERATION>>();

int main(int argc, char* argv[])
{
    primbench::settings settings;
    settings.size                 = 128 * primbench::MiB;
    settings.min_gpu_ms_per_batch = 100;
    primbench::executor executor(argc, argv, settings);

    // TODO: Make this file around ~500 lines shorter by queueing the benchmarks using nested macros

    // TODO: Split this "memory" file into subfiles, like benchmark_device_block_load_direct_blocked_vectorized.cpp

    // simple memory copy not running kernel
    CREATE_BENCHMARK(copy, no_operation, int32_t, 1, 1)

    // simple memory copy
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, int32_t, 128, 1)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, int32_t, 128, 2)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, int32_t, 128, 4)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, int32_t, 128, 8)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, int32_t, 128, 16)

    CREATE_BENCHMARK(block_primitives_transpose, no_operation, int32_t, 256, 1)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, int32_t, 256, 2)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, int32_t, 256, 4)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, int32_t, 256, 8)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, int32_t, 256, 16)

    CREATE_BENCHMARK(block_primitives_transpose, no_operation, int32_t, 512, 1)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, int32_t, 512, 2)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, int32_t, 512, 4)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, int32_t, 512, 8)

    CREATE_BENCHMARK(block_primitives_transpose, no_operation, int32_t, 1024, 1)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, int32_t, 1024, 2)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, int32_t, 1024, 4)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, int32_t, 1024, 8)

    CREATE_BENCHMARK(block_primitives_transpose, no_operation, uint64_t, 128, 1)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, uint64_t, 128, 2)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, uint64_t, 128, 4)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, uint64_t, 128, 8)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, uint64_t, 128, 16)

    CREATE_BENCHMARK(block_primitives_transpose, no_operation, uint64_t, 256, 1)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, uint64_t, 256, 2)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, uint64_t, 256, 4)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, uint64_t, 256, 8)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, uint64_t, 256, 16)

    CREATE_BENCHMARK(block_primitives_transpose, no_operation, uint64_t, 512, 1)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, uint64_t, 512, 2)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, uint64_t, 512, 4)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, uint64_t, 512, 8)

    CREATE_BENCHMARK(block_primitives_transpose, no_operation, uint64_t, 1024, 1)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, uint64_t, 1024, 2)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, uint64_t, 1024, 4)

    CREATE_BENCHMARK(block_primitives_transpose, no_operation, rocprim::int128_t, 128, 1)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, rocprim::int128_t, 128, 2)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, rocprim::int128_t, 128, 4)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, rocprim::int128_t, 128, 8)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, rocprim::int128_t, 128, 16)

    CREATE_BENCHMARK(block_primitives_transpose, no_operation, rocprim::int128_t, 256, 1)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, rocprim::int128_t, 256, 2)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, rocprim::int128_t, 256, 4)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, rocprim::int128_t, 256, 8)

    CREATE_BENCHMARK(block_primitives_transpose, no_operation, rocprim::int128_t, 512, 1)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, rocprim::int128_t, 512, 2)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, rocprim::int128_t, 512, 4)

    CREATE_BENCHMARK(block_primitives_transpose, no_operation, rocprim::int128_t, 1024, 1)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, rocprim::int128_t, 1024, 2)

    CREATE_BENCHMARK(block_primitives_transpose, no_operation, rocprim::uint128_t, 128, 1)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, rocprim::uint128_t, 128, 2)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, rocprim::uint128_t, 128, 4)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, rocprim::uint128_t, 128, 8)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, rocprim::uint128_t, 128, 16)

    CREATE_BENCHMARK(block_primitives_transpose, no_operation, rocprim::uint128_t, 256, 1)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, rocprim::uint128_t, 256, 2)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, rocprim::uint128_t, 256, 4)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, rocprim::uint128_t, 256, 8)

    CREATE_BENCHMARK(block_primitives_transpose, no_operation, rocprim::uint128_t, 512, 1)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, rocprim::uint128_t, 512, 2)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, rocprim::uint128_t, 512, 4)

    CREATE_BENCHMARK(block_primitives_transpose, no_operation, rocprim::uint128_t, 1024, 1)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, rocprim::uint128_t, 1024, 2)

    // simple memory copy using vector type
    CREATE_BENCHMARK(vectorized, no_operation, int32_t, 128, 1)
    CREATE_BENCHMARK(vectorized, no_operation, int32_t, 128, 2)
    CREATE_BENCHMARK(vectorized, no_operation, int32_t, 128, 4)
    CREATE_BENCHMARK(vectorized, no_operation, int32_t, 128, 8)
    CREATE_BENCHMARK(vectorized, no_operation, int32_t, 128, 16)

    CREATE_BENCHMARK(vectorized, no_operation, int32_t, 256, 1)
    CREATE_BENCHMARK(vectorized, no_operation, int32_t, 256, 2)
    CREATE_BENCHMARK(vectorized, no_operation, int32_t, 256, 4)
    CREATE_BENCHMARK(vectorized, no_operation, int32_t, 256, 8)
    CREATE_BENCHMARK(vectorized, no_operation, int32_t, 256, 16)

    CREATE_BENCHMARK(vectorized, no_operation, int32_t, 512, 1)
    CREATE_BENCHMARK(vectorized, no_operation, int32_t, 512, 2)
    CREATE_BENCHMARK(vectorized, no_operation, int32_t, 512, 4)
    CREATE_BENCHMARK(vectorized, no_operation, int32_t, 512, 8)

    CREATE_BENCHMARK(vectorized, no_operation, int32_t, 1024, 1)
    CREATE_BENCHMARK(vectorized, no_operation, int32_t, 1024, 2)
    CREATE_BENCHMARK(vectorized, no_operation, int32_t, 1024, 4)
    CREATE_BENCHMARK(vectorized, no_operation, int32_t, 1024, 8)

    CREATE_BENCHMARK(vectorized, no_operation, uint64_t, 128, 1)
    CREATE_BENCHMARK(vectorized, no_operation, uint64_t, 128, 2)
    CREATE_BENCHMARK(vectorized, no_operation, uint64_t, 128, 4)
    CREATE_BENCHMARK(vectorized, no_operation, uint64_t, 128, 8)
    CREATE_BENCHMARK(vectorized, no_operation, uint64_t, 128, 16)

    CREATE_BENCHMARK(vectorized, no_operation, uint64_t, 256, 1)
    CREATE_BENCHMARK(vectorized, no_operation, uint64_t, 256, 2)
    CREATE_BENCHMARK(vectorized, no_operation, uint64_t, 256, 4)
    CREATE_BENCHMARK(vectorized, no_operation, uint64_t, 256, 8)
    CREATE_BENCHMARK(vectorized, no_operation, uint64_t, 256, 16)

    CREATE_BENCHMARK(vectorized, no_operation, uint64_t, 512, 1)
    CREATE_BENCHMARK(vectorized, no_operation, uint64_t, 512, 2)
    CREATE_BENCHMARK(vectorized, no_operation, uint64_t, 512, 4)
    CREATE_BENCHMARK(vectorized, no_operation, uint64_t, 512, 8)

    CREATE_BENCHMARK(vectorized, no_operation, uint64_t, 1024, 1)
    CREATE_BENCHMARK(vectorized, no_operation, uint64_t, 1024, 2)
    CREATE_BENCHMARK(vectorized, no_operation, uint64_t, 1024, 4)
    CREATE_BENCHMARK(vectorized, no_operation, uint64_t, 1024, 8)

    CREATE_BENCHMARK(vectorized, no_operation, rocprim::int128_t, 128, 1)
    CREATE_BENCHMARK(vectorized, no_operation, rocprim::int128_t, 128, 2)
    CREATE_BENCHMARK(vectorized, no_operation, rocprim::int128_t, 128, 4)
    CREATE_BENCHMARK(vectorized, no_operation, rocprim::int128_t, 128, 8)
    CREATE_BENCHMARK(vectorized, no_operation, rocprim::int128_t, 128, 16)

    CREATE_BENCHMARK(vectorized, no_operation, rocprim::int128_t, 256, 1)
    CREATE_BENCHMARK(vectorized, no_operation, rocprim::int128_t, 256, 2)
    CREATE_BENCHMARK(vectorized, no_operation, rocprim::int128_t, 256, 4)
    CREATE_BENCHMARK(vectorized, no_operation, rocprim::int128_t, 256, 8)
    CREATE_BENCHMARK(vectorized, no_operation, rocprim::int128_t, 256, 16)

    CREATE_BENCHMARK(vectorized, no_operation, rocprim::int128_t, 512, 1)
    CREATE_BENCHMARK(vectorized, no_operation, rocprim::int128_t, 512, 2)
    CREATE_BENCHMARK(vectorized, no_operation, rocprim::int128_t, 512, 4)
    CREATE_BENCHMARK(vectorized, no_operation, rocprim::int128_t, 512, 8)

    CREATE_BENCHMARK(vectorized, no_operation, rocprim::int128_t, 1024, 1)
    CREATE_BENCHMARK(vectorized, no_operation, rocprim::int128_t, 1024, 2)
    CREATE_BENCHMARK(vectorized, no_operation, rocprim::int128_t, 1024, 4)
    CREATE_BENCHMARK(vectorized, no_operation, rocprim::int128_t, 1024, 8)

    CREATE_BENCHMARK(vectorized, no_operation, rocprim::uint128_t, 128, 1)
    CREATE_BENCHMARK(vectorized, no_operation, rocprim::uint128_t, 128, 2)
    CREATE_BENCHMARK(vectorized, no_operation, rocprim::uint128_t, 128, 4)
    CREATE_BENCHMARK(vectorized, no_operation, rocprim::uint128_t, 128, 8)
    CREATE_BENCHMARK(vectorized, no_operation, rocprim::uint128_t, 128, 16)

    CREATE_BENCHMARK(vectorized, no_operation, rocprim::uint128_t, 256, 1)
    CREATE_BENCHMARK(vectorized, no_operation, rocprim::uint128_t, 256, 2)
    CREATE_BENCHMARK(vectorized, no_operation, rocprim::uint128_t, 256, 4)
    CREATE_BENCHMARK(vectorized, no_operation, rocprim::uint128_t, 256, 8)
    CREATE_BENCHMARK(vectorized, no_operation, rocprim::uint128_t, 256, 16)

    CREATE_BENCHMARK(vectorized, no_operation, rocprim::uint128_t, 512, 1)
    CREATE_BENCHMARK(vectorized, no_operation, rocprim::uint128_t, 512, 2)
    CREATE_BENCHMARK(vectorized, no_operation, rocprim::uint128_t, 512, 4)
    CREATE_BENCHMARK(vectorized, no_operation, rocprim::uint128_t, 512, 8)

    CREATE_BENCHMARK(vectorized, no_operation, rocprim::uint128_t, 1024, 1)
    CREATE_BENCHMARK(vectorized, no_operation, rocprim::uint128_t, 1024, 2)
    CREATE_BENCHMARK(vectorized, no_operation, rocprim::uint128_t, 1024, 4)
    CREATE_BENCHMARK(vectorized, no_operation, rocprim::uint128_t, 1024, 8)

    // simple memory copy using striped
    CREATE_BENCHMARK(striped, no_operation, int32_t, 128, 1)
    CREATE_BENCHMARK(striped, no_operation, int32_t, 128, 2)
    CREATE_BENCHMARK(striped, no_operation, int32_t, 128, 4)
    CREATE_BENCHMARK(striped, no_operation, int32_t, 128, 8)
    CREATE_BENCHMARK(striped, no_operation, int32_t, 128, 16)

    CREATE_BENCHMARK(striped, no_operation, int32_t, 256, 1)
    CREATE_BENCHMARK(striped, no_operation, int32_t, 256, 2)
    CREATE_BENCHMARK(striped, no_operation, int32_t, 256, 4)
    CREATE_BENCHMARK(striped, no_operation, int32_t, 256, 8)
    CREATE_BENCHMARK(striped, no_operation, int32_t, 256, 16)

    CREATE_BENCHMARK(striped, no_operation, int32_t, 512, 1)
    CREATE_BENCHMARK(striped, no_operation, int32_t, 512, 2)
    CREATE_BENCHMARK(striped, no_operation, int32_t, 512, 4)
    CREATE_BENCHMARK(striped, no_operation, int32_t, 512, 8)

    CREATE_BENCHMARK(striped, no_operation, int32_t, 1024, 1)
    CREATE_BENCHMARK(striped, no_operation, int32_t, 1024, 2)
    CREATE_BENCHMARK(striped, no_operation, int32_t, 1024, 4)
    CREATE_BENCHMARK(striped, no_operation, int32_t, 1024, 8)

    CREATE_BENCHMARK(striped, no_operation, uint64_t, 128, 1)
    CREATE_BENCHMARK(striped, no_operation, uint64_t, 128, 2)
    CREATE_BENCHMARK(striped, no_operation, uint64_t, 128, 4)
    CREATE_BENCHMARK(striped, no_operation, uint64_t, 128, 8)
    CREATE_BENCHMARK(striped, no_operation, uint64_t, 128, 16)

    CREATE_BENCHMARK(striped, no_operation, uint64_t, 256, 1)
    CREATE_BENCHMARK(striped, no_operation, uint64_t, 256, 2)
    CREATE_BENCHMARK(striped, no_operation, uint64_t, 256, 4)
    CREATE_BENCHMARK(striped, no_operation, uint64_t, 256, 8)
    CREATE_BENCHMARK(striped, no_operation, uint64_t, 256, 16)

    CREATE_BENCHMARK(striped, no_operation, uint64_t, 512, 1)
    CREATE_BENCHMARK(striped, no_operation, uint64_t, 512, 2)
    CREATE_BENCHMARK(striped, no_operation, uint64_t, 512, 4)
    CREATE_BENCHMARK(striped, no_operation, uint64_t, 512, 8)

    CREATE_BENCHMARK(striped, no_operation, uint64_t, 1024, 1)
    CREATE_BENCHMARK(striped, no_operation, uint64_t, 1024, 2)
    CREATE_BENCHMARK(striped, no_operation, uint64_t, 1024, 4)
    CREATE_BENCHMARK(striped, no_operation, uint64_t, 1024, 8)

    CREATE_BENCHMARK(striped, no_operation, rocprim::int128_t, 128, 1)
    CREATE_BENCHMARK(striped, no_operation, rocprim::int128_t, 128, 2)
    CREATE_BENCHMARK(striped, no_operation, rocprim::int128_t, 128, 4)
    CREATE_BENCHMARK(striped, no_operation, rocprim::int128_t, 128, 8)
    CREATE_BENCHMARK(striped, no_operation, rocprim::int128_t, 128, 16)

    CREATE_BENCHMARK(striped, no_operation, rocprim::int128_t, 256, 1)
    CREATE_BENCHMARK(striped, no_operation, rocprim::int128_t, 256, 2)
    CREATE_BENCHMARK(striped, no_operation, rocprim::int128_t, 256, 4)
    CREATE_BENCHMARK(striped, no_operation, rocprim::int128_t, 256, 8)
    CREATE_BENCHMARK(striped, no_operation, rocprim::int128_t, 256, 16)

    CREATE_BENCHMARK(striped, no_operation, rocprim::int128_t, 512, 1)
    CREATE_BENCHMARK(striped, no_operation, rocprim::int128_t, 512, 2)
    CREATE_BENCHMARK(striped, no_operation, rocprim::int128_t, 512, 4)
    CREATE_BENCHMARK(striped, no_operation, rocprim::int128_t, 512, 8)

    CREATE_BENCHMARK(striped, no_operation, rocprim::int128_t, 1024, 1)
    CREATE_BENCHMARK(striped, no_operation, rocprim::int128_t, 1024, 2)
    CREATE_BENCHMARK(striped, no_operation, rocprim::int128_t, 1024, 4)
    CREATE_BENCHMARK(striped, no_operation, rocprim::int128_t, 1024, 8)

    CREATE_BENCHMARK(striped, no_operation, rocprim::uint128_t, 128, 1)
    CREATE_BENCHMARK(striped, no_operation, rocprim::uint128_t, 128, 2)
    CREATE_BENCHMARK(striped, no_operation, rocprim::uint128_t, 128, 4)
    CREATE_BENCHMARK(striped, no_operation, rocprim::uint128_t, 128, 8)
    CREATE_BENCHMARK(striped, no_operation, rocprim::uint128_t, 128, 16)

    CREATE_BENCHMARK(striped, no_operation, rocprim::uint128_t, 256, 1)
    CREATE_BENCHMARK(striped, no_operation, rocprim::uint128_t, 256, 2)
    CREATE_BENCHMARK(striped, no_operation, rocprim::uint128_t, 256, 4)
    CREATE_BENCHMARK(striped, no_operation, rocprim::uint128_t, 256, 8)
    CREATE_BENCHMARK(striped, no_operation, rocprim::uint128_t, 256, 16)

    CREATE_BENCHMARK(striped, no_operation, rocprim::uint128_t, 512, 1)
    CREATE_BENCHMARK(striped, no_operation, rocprim::uint128_t, 512, 2)
    CREATE_BENCHMARK(striped, no_operation, rocprim::uint128_t, 512, 4)
    CREATE_BENCHMARK(striped, no_operation, rocprim::uint128_t, 512, 8)

    CREATE_BENCHMARK(striped, no_operation, rocprim::uint128_t, 1024, 1)
    CREATE_BENCHMARK(striped, no_operation, rocprim::uint128_t, 1024, 2)
    CREATE_BENCHMARK(striped, no_operation, rocprim::uint128_t, 1024, 4)
    CREATE_BENCHMARK(striped, no_operation, rocprim::uint128_t, 1024, 8)

    // block_scan
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, int32_t, 128, 1)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, int32_t, 128, 2)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, int32_t, 128, 4)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, int32_t, 128, 8)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, int32_t, 128, 16)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, int32_t, 128, 32)

    CREATE_BENCHMARK(block_primitives_transpose, block_scan, int32_t, 256, 1)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, int32_t, 256, 2)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, int32_t, 256, 4)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, int32_t, 256, 8)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, int32_t, 256, 16)

    CREATE_BENCHMARK(block_primitives_transpose, block_scan, int32_t, 512, 1)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, int32_t, 512, 2)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, int32_t, 512, 4)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, int32_t, 512, 8)

    CREATE_BENCHMARK(block_primitives_transpose, block_scan, int32_t, 1024, 1)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, int32_t, 1024, 2)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, int32_t, 1024, 4)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, int32_t, 1024, 8)

    CREATE_BENCHMARK(block_primitives_transpose, block_scan, float, 256, 1)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, float, 256, 2)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, float, 256, 4)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, float, 256, 8)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, float, 256, 16)

    CREATE_BENCHMARK(block_primitives_transpose, block_scan, float, 512, 1)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, float, 512, 2)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, float, 512, 4)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, float, 512, 8)

    CREATE_BENCHMARK(block_primitives_transpose, block_scan, float, 1024, 1)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, float, 1024, 2)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, float, 1024, 4)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, float, 1024, 8)

    CREATE_BENCHMARK(block_primitives_transpose, block_scan, double, 128, 1)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, double, 128, 2)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, double, 128, 4)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, double, 128, 8)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, double, 128, 16)

    CREATE_BENCHMARK(block_primitives_transpose, block_scan, double, 256, 1)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, double, 256, 2)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, double, 256, 4)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, double, 256, 8)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, double, 256, 16)

    CREATE_BENCHMARK(block_primitives_transpose, block_scan, double, 512, 1)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, double, 512, 2)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, double, 512, 4)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, double, 512, 8)

    CREATE_BENCHMARK(block_primitives_transpose, block_scan, double, 1024, 1)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, double, 1024, 2)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, double, 1024, 4)

    CREATE_BENCHMARK(block_primitives_transpose, block_scan, uint64_t, 128, 1)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, uint64_t, 128, 2)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, uint64_t, 128, 4)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, uint64_t, 128, 8)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, uint64_t, 128, 16)

    CREATE_BENCHMARK(block_primitives_transpose, block_scan, uint64_t, 256, 1)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, uint64_t, 256, 2)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, uint64_t, 256, 4)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, uint64_t, 256, 8)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, uint64_t, 256, 16)

    CREATE_BENCHMARK(block_primitives_transpose, block_scan, uint64_t, 512, 1)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, uint64_t, 512, 2)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, uint64_t, 512, 4)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, uint64_t, 512, 8)

    CREATE_BENCHMARK(block_primitives_transpose, block_scan, uint64_t, 1024, 1)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, uint64_t, 1024, 2)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, uint64_t, 1024, 4)

    CREATE_BENCHMARK(block_primitives_transpose, block_scan, rocprim::int128_t, 128, 1)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, rocprim::int128_t, 128, 2)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, rocprim::int128_t, 128, 4)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, rocprim::int128_t, 128, 8)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, rocprim::int128_t, 128, 16)

    CREATE_BENCHMARK(block_primitives_transpose, block_scan, rocprim::int128_t, 256, 1)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, rocprim::int128_t, 256, 2)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, rocprim::int128_t, 256, 4)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, rocprim::int128_t, 256, 8)

    CREATE_BENCHMARK(block_primitives_transpose, block_scan, rocprim::int128_t, 512, 1)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, rocprim::int128_t, 512, 2)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, rocprim::int128_t, 512, 4)

    CREATE_BENCHMARK(block_primitives_transpose, block_scan, rocprim::int128_t, 1024, 1)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, rocprim::int128_t, 1024, 2)

    CREATE_BENCHMARK(block_primitives_transpose, block_scan, rocprim::uint128_t, 128, 1)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, rocprim::uint128_t, 128, 2)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, rocprim::uint128_t, 128, 4)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, rocprim::uint128_t, 128, 8)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, rocprim::uint128_t, 128, 16)

    CREATE_BENCHMARK(block_primitives_transpose, block_scan, rocprim::uint128_t, 256, 1)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, rocprim::uint128_t, 256, 2)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, rocprim::uint128_t, 256, 4)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, rocprim::uint128_t, 256, 8)

    CREATE_BENCHMARK(block_primitives_transpose, block_scan, rocprim::uint128_t, 512, 1)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, rocprim::uint128_t, 512, 2)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, rocprim::uint128_t, 512, 4)

    CREATE_BENCHMARK(block_primitives_transpose, block_scan, rocprim::uint128_t, 1024, 1)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, rocprim::uint128_t, 1024, 2)

    // vectorized - block_scan
    CREATE_BENCHMARK(vectorized, block_scan, int32_t, 128, 1)
    CREATE_BENCHMARK(vectorized, block_scan, int32_t, 128, 2)
    CREATE_BENCHMARK(vectorized, block_scan, int32_t, 128, 4)
    CREATE_BENCHMARK(vectorized, block_scan, int32_t, 128, 8)
    CREATE_BENCHMARK(vectorized, block_scan, int32_t, 128, 16)

    CREATE_BENCHMARK(vectorized, block_scan, int32_t, 256, 1)
    CREATE_BENCHMARK(vectorized, block_scan, int32_t, 256, 2)
    CREATE_BENCHMARK(vectorized, block_scan, int32_t, 256, 4)
    CREATE_BENCHMARK(vectorized, block_scan, int32_t, 256, 8)
    CREATE_BENCHMARK(vectorized, block_scan, int32_t, 256, 16)

    CREATE_BENCHMARK(vectorized, block_scan, int32_t, 512, 1)
    CREATE_BENCHMARK(vectorized, block_scan, int32_t, 512, 2)
    CREATE_BENCHMARK(vectorized, block_scan, int32_t, 512, 4)
    CREATE_BENCHMARK(vectorized, block_scan, int32_t, 512, 8)

    CREATE_BENCHMARK(vectorized, block_scan, int32_t, 1024, 1)
    CREATE_BENCHMARK(vectorized, block_scan, int32_t, 1024, 2)
    CREATE_BENCHMARK(vectorized, block_scan, int32_t, 1024, 4)
    CREATE_BENCHMARK(vectorized, block_scan, int32_t, 1024, 8)

    CREATE_BENCHMARK(vectorized, block_scan, float, 128, 1)
    CREATE_BENCHMARK(vectorized, block_scan, float, 128, 2)
    CREATE_BENCHMARK(vectorized, block_scan, float, 128, 4)
    CREATE_BENCHMARK(vectorized, block_scan, float, 128, 8)
    CREATE_BENCHMARK(vectorized, block_scan, float, 128, 16)

    CREATE_BENCHMARK(vectorized, block_scan, float, 256, 1)
    CREATE_BENCHMARK(vectorized, block_scan, float, 256, 2)
    CREATE_BENCHMARK(vectorized, block_scan, float, 256, 4)
    CREATE_BENCHMARK(vectorized, block_scan, float, 256, 8)
    CREATE_BENCHMARK(vectorized, block_scan, float, 256, 16)

    CREATE_BENCHMARK(vectorized, block_scan, float, 512, 1)
    CREATE_BENCHMARK(vectorized, block_scan, float, 512, 2)
    CREATE_BENCHMARK(vectorized, block_scan, float, 512, 4)
    CREATE_BENCHMARK(vectorized, block_scan, float, 512, 8)

    CREATE_BENCHMARK(vectorized, block_scan, float, 1024, 1)
    CREATE_BENCHMARK(vectorized, block_scan, float, 1024, 2)
    CREATE_BENCHMARK(vectorized, block_scan, float, 1024, 4)
    CREATE_BENCHMARK(vectorized, block_scan, float, 1024, 8)

    CREATE_BENCHMARK(vectorized, block_scan, double, 128, 1)
    CREATE_BENCHMARK(vectorized, block_scan, double, 128, 2)
    CREATE_BENCHMARK(vectorized, block_scan, double, 128, 4)
    CREATE_BENCHMARK(vectorized, block_scan, double, 128, 8)
    CREATE_BENCHMARK(vectorized, block_scan, double, 128, 16)

    CREATE_BENCHMARK(vectorized, block_scan, double, 256, 1)
    CREATE_BENCHMARK(vectorized, block_scan, double, 256, 2)
    CREATE_BENCHMARK(vectorized, block_scan, double, 256, 4)
    CREATE_BENCHMARK(vectorized, block_scan, double, 256, 8)
    CREATE_BENCHMARK(vectorized, block_scan, double, 256, 16)

    CREATE_BENCHMARK(vectorized, block_scan, double, 512, 1)
    CREATE_BENCHMARK(vectorized, block_scan, double, 512, 2)
    CREATE_BENCHMARK(vectorized, block_scan, double, 512, 4)
    CREATE_BENCHMARK(vectorized, block_scan, double, 512, 8)

    CREATE_BENCHMARK(vectorized, block_scan, double, 1024, 1)
    CREATE_BENCHMARK(vectorized, block_scan, double, 1024, 2)
    CREATE_BENCHMARK(vectorized, block_scan, double, 1024, 4)

    CREATE_BENCHMARK(vectorized, block_scan, uint64_t, 128, 1)
    CREATE_BENCHMARK(vectorized, block_scan, uint64_t, 128, 2)
    CREATE_BENCHMARK(vectorized, block_scan, uint64_t, 128, 4)
    CREATE_BENCHMARK(vectorized, block_scan, uint64_t, 128, 8)
    CREATE_BENCHMARK(vectorized, block_scan, uint64_t, 128, 16)

    CREATE_BENCHMARK(vectorized, block_scan, uint64_t, 256, 1)
    CREATE_BENCHMARK(vectorized, block_scan, uint64_t, 256, 2)
    CREATE_BENCHMARK(vectorized, block_scan, uint64_t, 256, 4)
    CREATE_BENCHMARK(vectorized, block_scan, uint64_t, 256, 8)
    CREATE_BENCHMARK(vectorized, block_scan, uint64_t, 256, 16)

    CREATE_BENCHMARK(vectorized, block_scan, uint64_t, 512, 1)
    CREATE_BENCHMARK(vectorized, block_scan, uint64_t, 512, 2)
    CREATE_BENCHMARK(vectorized, block_scan, uint64_t, 512, 4)
    CREATE_BENCHMARK(vectorized, block_scan, uint64_t, 512, 8)

    CREATE_BENCHMARK(vectorized, block_scan, uint64_t, 1024, 1)
    CREATE_BENCHMARK(vectorized, block_scan, uint64_t, 1024, 2)
    CREATE_BENCHMARK(vectorized, block_scan, uint64_t, 1024, 4)

    CREATE_BENCHMARK(vectorized, block_scan, rocprim::int128_t, 128, 1)
    CREATE_BENCHMARK(vectorized, block_scan, rocprim::int128_t, 128, 2)
    CREATE_BENCHMARK(vectorized, block_scan, rocprim::int128_t, 128, 4)
    CREATE_BENCHMARK(vectorized, block_scan, rocprim::int128_t, 128, 8)
    CREATE_BENCHMARK(vectorized, block_scan, rocprim::int128_t, 128, 16)

    CREATE_BENCHMARK(vectorized, block_scan, rocprim::int128_t, 256, 1)
    CREATE_BENCHMARK(vectorized, block_scan, rocprim::int128_t, 256, 2)
    CREATE_BENCHMARK(vectorized, block_scan, rocprim::int128_t, 256, 4)
    CREATE_BENCHMARK(vectorized, block_scan, rocprim::int128_t, 256, 8)
    CREATE_BENCHMARK(vectorized, block_scan, rocprim::int128_t, 256, 16)

    CREATE_BENCHMARK(vectorized, block_scan, rocprim::int128_t, 512, 1)
    CREATE_BENCHMARK(vectorized, block_scan, rocprim::int128_t, 512, 2)
    CREATE_BENCHMARK(vectorized, block_scan, rocprim::int128_t, 512, 4)
    CREATE_BENCHMARK(vectorized, block_scan, rocprim::int128_t, 512, 8)

    CREATE_BENCHMARK(vectorized, block_scan, rocprim::int128_t, 1024, 1)
    CREATE_BENCHMARK(vectorized, block_scan, rocprim::int128_t, 1024, 2)
    CREATE_BENCHMARK(vectorized, block_scan, rocprim::int128_t, 1024, 4)

    CREATE_BENCHMARK(vectorized, block_scan, rocprim::uint128_t, 128, 1)
    CREATE_BENCHMARK(vectorized, block_scan, rocprim::uint128_t, 128, 2)
    CREATE_BENCHMARK(vectorized, block_scan, rocprim::uint128_t, 128, 4)
    CREATE_BENCHMARK(vectorized, block_scan, rocprim::uint128_t, 128, 8)
    CREATE_BENCHMARK(vectorized, block_scan, rocprim::uint128_t, 128, 16)

    CREATE_BENCHMARK(vectorized, block_scan, rocprim::uint128_t, 256, 1)
    CREATE_BENCHMARK(vectorized, block_scan, rocprim::uint128_t, 256, 2)
    CREATE_BENCHMARK(vectorized, block_scan, rocprim::uint128_t, 256, 4)
    CREATE_BENCHMARK(vectorized, block_scan, rocprim::uint128_t, 256, 8)
    CREATE_BENCHMARK(vectorized, block_scan, rocprim::uint128_t, 256, 16)

    CREATE_BENCHMARK(vectorized, block_scan, rocprim::uint128_t, 512, 1)
    CREATE_BENCHMARK(vectorized, block_scan, rocprim::uint128_t, 512, 2)
    CREATE_BENCHMARK(vectorized, block_scan, rocprim::uint128_t, 512, 4)
    CREATE_BENCHMARK(vectorized, block_scan, rocprim::uint128_t, 512, 8)

    CREATE_BENCHMARK(vectorized, block_scan, rocprim::uint128_t, 1024, 1)
    CREATE_BENCHMARK(vectorized, block_scan, rocprim::uint128_t, 1024, 2)
    CREATE_BENCHMARK(vectorized, block_scan, rocprim::uint128_t, 1024, 4)

    // custom_op
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, int32_t, 128, 1)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, int32_t, 128, 2)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, int32_t, 128, 4)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, int32_t, 128, 8)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, int32_t, 128, 16)

    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, int32_t, 256, 1)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, int32_t, 256, 2)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, int32_t, 256, 4)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, int32_t, 256, 8)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, int32_t, 256, 16)

    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, int32_t, 512, 1)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, int32_t, 512, 2)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, int32_t, 512, 4)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, int32_t, 512, 8)

    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, int32_t, 1024, 1)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, int32_t, 1024, 2)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, int32_t, 1024, 4)

    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, float, 128, 1)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, float, 128, 2)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, float, 128, 4)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, float, 128, 8)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, float, 128, 16)

    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, float, 256, 1)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, float, 256, 2)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, float, 256, 4)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, float, 256, 8)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, float, 256, 16)

    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, float, 512, 1)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, float, 512, 2)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, float, 512, 4)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, float, 512, 8)

    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, float, 1024, 1)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, float, 1024, 2)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, float, 1024, 4)

    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, double, 128, 1)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, double, 128, 2)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, double, 128, 4)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, double, 128, 8)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, double, 128, 16)

    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, double, 256, 1)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, double, 256, 2)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, double, 256, 4)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, double, 256, 8)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, double, 256, 16)

    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, double, 512, 1)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, double, 512, 2)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, double, 512, 4)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, double, 512, 8)

    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, double, 1024, 1)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, double, 1024, 2)

    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, uint64_t, 128, 1)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, uint64_t, 128, 2)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, uint64_t, 128, 4)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, uint64_t, 128, 8)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, uint64_t, 128, 16)

    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, uint64_t, 256, 1)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, uint64_t, 256, 2)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, uint64_t, 256, 4)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, uint64_t, 256, 8)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, uint64_t, 256, 16)

    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, uint64_t, 512, 1)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, uint64_t, 512, 2)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, uint64_t, 512, 4)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, uint64_t, 512, 8)

    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, uint64_t, 1024, 1)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, uint64_t, 1024, 2)

    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, rocprim::int128_t, 128, 1)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, rocprim::int128_t, 128, 2)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, rocprim::int128_t, 128, 4)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, rocprim::int128_t, 128, 8)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, rocprim::int128_t, 128, 16)

    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, rocprim::int128_t, 256, 1)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, rocprim::int128_t, 256, 2)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, rocprim::int128_t, 256, 4)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, rocprim::int128_t, 256, 8)

    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, rocprim::int128_t, 512, 1)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, rocprim::int128_t, 512, 2)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, rocprim::int128_t, 512, 4)

    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, rocprim::int128_t, 1024, 1)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, rocprim::int128_t, 1024, 2)

    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, rocprim::uint128_t, 128, 1)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, rocprim::uint128_t, 128, 2)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, rocprim::uint128_t, 128, 4)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, rocprim::uint128_t, 128, 8)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, rocprim::uint128_t, 128, 16)

    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, rocprim::uint128_t, 256, 1)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, rocprim::uint128_t, 256, 2)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, rocprim::uint128_t, 256, 4)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, rocprim::uint128_t, 256, 8)

    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, rocprim::uint128_t, 512, 1)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, rocprim::uint128_t, 512, 2)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, rocprim::uint128_t, 512, 4)

    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, rocprim::uint128_t, 1024, 1)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, rocprim::uint128_t, 1024, 2)

    // block_primitives_transpose - atomics no collision
    CREATE_BENCHMARK(block_primitives_transpose, atomics_no_collision, int32_t, 128, 1)
    CREATE_BENCHMARK(block_primitives_transpose, atomics_no_collision, int32_t, 128, 2)
    CREATE_BENCHMARK(block_primitives_transpose, atomics_no_collision, int32_t, 128, 4)
    CREATE_BENCHMARK(block_primitives_transpose, atomics_no_collision, int32_t, 128, 8)
    CREATE_BENCHMARK(block_primitives_transpose, atomics_no_collision, int32_t, 128, 16)

    CREATE_BENCHMARK(block_primitives_transpose, atomics_no_collision, int32_t, 256, 1)
    CREATE_BENCHMARK(block_primitives_transpose, atomics_no_collision, int32_t, 256, 2)
    CREATE_BENCHMARK(block_primitives_transpose, atomics_no_collision, int32_t, 256, 4)
    CREATE_BENCHMARK(block_primitives_transpose, atomics_no_collision, int32_t, 256, 8)
    CREATE_BENCHMARK(block_primitives_transpose, atomics_no_collision, int32_t, 256, 16)

    CREATE_BENCHMARK(block_primitives_transpose, atomics_no_collision, int32_t, 512, 1)
    CREATE_BENCHMARK(block_primitives_transpose, atomics_no_collision, int32_t, 512, 2)
    CREATE_BENCHMARK(block_primitives_transpose, atomics_no_collision, int32_t, 512, 4)
    CREATE_BENCHMARK(block_primitives_transpose, atomics_no_collision, int32_t, 512, 8)

    CREATE_BENCHMARK(block_primitives_transpose, atomics_no_collision, int32_t, 1024, 1)
    CREATE_BENCHMARK(block_primitives_transpose, atomics_no_collision, int32_t, 1024, 2)
    CREATE_BENCHMARK(block_primitives_transpose, atomics_no_collision, int32_t, 1024, 4)
    CREATE_BENCHMARK(block_primitives_transpose, atomics_no_collision, int32_t, 1024, 8)

    // block_primitives_transpose - atomics inter block collision
    CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_block_collision, int32_t, 128, 1)
    CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_block_collision, int32_t, 128, 2)
    CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_block_collision, int32_t, 128, 4)
    CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_block_collision, int32_t, 128, 8)
    CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_block_collision, int32_t, 128, 16)

    CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_block_collision, int32_t, 256, 1)
    CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_block_collision, int32_t, 256, 2)
    CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_block_collision, int32_t, 256, 4)
    CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_block_collision, int32_t, 256, 8)
    CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_block_collision, int32_t, 256, 16)

    CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_block_collision, int32_t, 512, 1)
    CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_block_collision, int32_t, 512, 2)
    CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_block_collision, int32_t, 512, 4)
    CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_block_collision, int32_t, 512, 8)

    CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_block_collision, int32_t, 1024, 1)
    CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_block_collision, int32_t, 1024, 2)
    CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_block_collision, int32_t, 1024, 4)
    CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_block_collision, int32_t, 1024, 8)

    // block_primitives_transpose - atomics inter warp collision
    CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_warp_collision, int32_t, 128, 1)
    CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_warp_collision, int32_t, 128, 2)
    CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_warp_collision, int32_t, 128, 4)
    CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_warp_collision, int32_t, 128, 8)
    CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_warp_collision, int32_t, 128, 16)

    CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_warp_collision, int32_t, 256, 1)
    CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_warp_collision, int32_t, 256, 2)
    CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_warp_collision, int32_t, 256, 4)
    CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_warp_collision, int32_t, 256, 8)
    CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_warp_collision, int32_t, 256, 16)

    CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_warp_collision, int32_t, 512, 1)
    CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_warp_collision, int32_t, 512, 2)
    CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_warp_collision, int32_t, 512, 4)
    CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_warp_collision, int32_t, 512, 8)

    CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_warp_collision, int32_t, 1024, 1)
    CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_warp_collision, int32_t, 1024, 2)
    CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_warp_collision, int32_t, 1024, 4)
    CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_warp_collision, int32_t, 1024, 8)

    executor.run();
}
