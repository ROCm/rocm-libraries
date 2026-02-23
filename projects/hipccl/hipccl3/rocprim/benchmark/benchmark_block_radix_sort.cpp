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

template<benchmark_kinds BenchmarkKind>
void add_benchmarks(primbench::executor& executor)
{
    using custom_int_type = common::custom_type<int32_t, int32_t>;

    BENCHMARK_TYPE(int32_t, 64, 3)
    BENCHMARK_TYPE(int32_t, 512, 3)

    BENCHMARK_TYPE(int32_t, 64, 4)
    BENCHMARK_TYPE(int32_t, 128, 4)
    BENCHMARK_TYPE(int32_t, 192, 4)
    BENCHMARK_TYPE(int32_t, 256, 4)
    BENCHMARK_TYPE(int32_t, 320, 4)
    BENCHMARK_TYPE(int32_t, 512, 4)

    BENCHMARK_TYPE(int8_t, 64, 3)
    BENCHMARK_TYPE(int8_t, 512, 3)

    BENCHMARK_TYPE(int8_t, 64, 4)
    BENCHMARK_TYPE(int8_t, 128, 4)
    BENCHMARK_TYPE(int8_t, 192, 4)
    BENCHMARK_TYPE(int8_t, 256, 4)
    BENCHMARK_TYPE(int8_t, 320, 4)
    BENCHMARK_TYPE(int8_t, 512, 4)

    BENCHMARK_TYPE(uint8_t, 64, 3)
    BENCHMARK_TYPE(uint8_t, 512, 3)

    BENCHMARK_TYPE(uint8_t, 64, 4)
    BENCHMARK_TYPE(uint8_t, 128, 4)
    BENCHMARK_TYPE(uint8_t, 192, 4)
    BENCHMARK_TYPE(uint8_t, 256, 4)
    BENCHMARK_TYPE(uint8_t, 320, 4)
    BENCHMARK_TYPE(uint8_t, 512, 4)

    BENCHMARK_TYPE(rocprim::half, 64, 3)
    BENCHMARK_TYPE(rocprim::half, 512, 3)

    BENCHMARK_TYPE(rocprim::half, 64, 4)
    BENCHMARK_TYPE(rocprim::half, 128, 4)
    BENCHMARK_TYPE(rocprim::half, 192, 4)
    BENCHMARK_TYPE(rocprim::half, 256, 4)
    BENCHMARK_TYPE(rocprim::half, 320, 4)
    BENCHMARK_TYPE(rocprim::half, 512, 4)

    BENCHMARK_TYPE(int64_t, 64, 3)
    BENCHMARK_TYPE(int64_t, 512, 3)

    BENCHMARK_TYPE(int64_t, 64, 4)
    BENCHMARK_TYPE(int64_t, 128, 4)
    BENCHMARK_TYPE(int64_t, 192, 4)
    BENCHMARK_TYPE(int64_t, 256, 4)
    BENCHMARK_TYPE(int64_t, 320, 4)
    BENCHMARK_TYPE(int64_t, 512, 4)

    BENCHMARK_TYPE(custom_int_type, 64, 3)
    BENCHMARK_TYPE(custom_int_type, 512, 3)

    BENCHMARK_TYPE(custom_int_type, 64, 4)
    BENCHMARK_TYPE(custom_int_type, 128, 4)
    BENCHMARK_TYPE(custom_int_type, 192, 4)
    BENCHMARK_TYPE(custom_int_type, 256, 4)
    BENCHMARK_TYPE(custom_int_type, 320, 4)
    BENCHMARK_TYPE(custom_int_type, 512, 4)

    BENCHMARK_TYPE(rocprim::int128_t, 64, 3)
    BENCHMARK_TYPE(rocprim::int128_t, 512, 3)

    BENCHMARK_TYPE(rocprim::int128_t, 64, 4)
    BENCHMARK_TYPE(rocprim::int128_t, 128, 4)
    BENCHMARK_TYPE(rocprim::int128_t, 192, 4)
    BENCHMARK_TYPE(rocprim::int128_t, 256, 4)
    BENCHMARK_TYPE(rocprim::int128_t, 320, 4)
    BENCHMARK_TYPE(rocprim::int128_t, 512, 4)

    BENCHMARK_TYPE(rocprim::uint128_t, 64, 3)
    BENCHMARK_TYPE(rocprim::uint128_t, 512, 3)

    BENCHMARK_TYPE(rocprim::uint128_t, 64, 4)
    BENCHMARK_TYPE(rocprim::uint128_t, 128, 4)
    BENCHMARK_TYPE(rocprim::uint128_t, 192, 4)
    BENCHMARK_TYPE(rocprim::uint128_t, 256, 4)
    BENCHMARK_TYPE(rocprim::uint128_t, 320, 4)
    BENCHMARK_TYPE(rocprim::uint128_t, 512, 4)
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
