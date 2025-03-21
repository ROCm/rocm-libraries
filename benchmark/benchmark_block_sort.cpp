// MIT License
//
// Copyright (c) 2017-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "benchmark_block_sort.parallel.hpp"

// HIP API
#include <hip/hip_runtime.h>

// rocPRIM
#ifndef BENCHMARK_CONFIG_TUNING
    #include <rocprim/block/block_sort.hpp>
    #include <rocprim/types.hpp>
#endif

#include <cstddef>
#include <string>
#include <vector>
#ifndef BENCHMARK_CONFIG_TUNING
    #include <stdint.h>
#endif

#define CREATE_BENCHMARK_IPT_ALG(K, V, BS, IPT, ALG)       \
    benchmark_utils::executor::queue_sorted_instance<      \
        block_sort_benchmark<K, V, BS, IPT, ALG, true>>(); \
    benchmark_utils::executor::queue_sorted_instance<      \
        block_sort_benchmark<K, V, BS, IPT, ALG, false>>();

#define CREATE_BENCHMARK_IPT(K, V, BS, IPT)                                                   \
    CREATE_BENCHMARK_IPT_ALG(K, V, BS, IPT, rocprim::block_sort_algorithm::merge_sort)        \
    CREATE_BENCHMARK_IPT_ALG(K, V, BS, IPT, rocprim::block_sort_algorithm::stable_merge_sort) \
    CREATE_BENCHMARK_IPT_ALG(K, V, BS, IPT, rocprim::block_sort_algorithm::bitonic_sort)

#define CREATE_BENCHMARK(K, V, BS)    \
    CREATE_BENCHMARK_IPT(K, V, BS, 1) \
    CREATE_BENCHMARK_IPT(K, V, BS, 4)

int main(int argc, char* argv[])
{
    benchmark_utils::executor executor(argc, argv, 512 * benchmark_utils::MiB, 10, 0);

    // Block sizes as large as possible are most relevant
    CREATE_BENCHMARK(float, rocprim::empty_type, 256)
    CREATE_BENCHMARK(double, rocprim::empty_type, 256)
    CREATE_BENCHMARK(rocprim::half, rocprim::empty_type, 256)
    CREATE_BENCHMARK(uint8_t, rocprim::empty_type, 256)
    CREATE_BENCHMARK(int, rocprim::empty_type, 256)
    CREATE_BENCHMARK(int, rocprim::empty_type, 512)
    CREATE_BENCHMARK(double, rocprim::empty_type, 512)
    CREATE_BENCHMARK(rocprim::int128_t, rocprim::empty_type, 256)
    CREATE_BENCHMARK(rocprim::uint128_t, rocprim::empty_type, 256)
    CREATE_BENCHMARK(int, int, 512)
    CREATE_BENCHMARK(float, double, 512)
    CREATE_BENCHMARK(double, int64_t, 512)
    CREATE_BENCHMARK(rocprim::half, int16_t, 512)
    CREATE_BENCHMARK(uint8_t, uint32_t, 512)
    CREATE_BENCHMARK(int64_t, rocprim::int128_t, 512)
    CREATE_BENCHMARK(uint64_t, rocprim::uint128_t, 512)

    executor.run();
}
