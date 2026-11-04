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

#include "benchmark_block_radix_rank.hpp"

#include "primbench.hpp"

#define CREATE_BENCHMARK(T, BS, IPT, ALGO) \
    executor.queue<block_radix_rank_benchmark<T, BS, IPT, ALGO>>();

// clang-format off
#define CREATE_BENCHMARK_KINDS(type, block, ipt) \
    CREATE_BENCHMARK(type, block, ipt, rocprim::block_radix_rank_algorithm::basic) \
    CREATE_BENCHMARK(type, block, ipt, rocprim::block_radix_rank_algorithm::basic_memoize) \
    CREATE_BENCHMARK(type, block, ipt, rocprim::block_radix_rank_algorithm::match)

#define BENCHMARK_TYPE(type, block) \
    CREATE_BENCHMARK_KINDS(type, block, 1) \
    CREATE_BENCHMARK_KINDS(type, block, 4) \
    CREATE_BENCHMARK_KINDS(type, block, 8) \
    CREATE_BENCHMARK_KINDS(type, block, 12) \
    CREATE_BENCHMARK_KINDS(type, block, 16) \
    CREATE_BENCHMARK_KINDS(type, block, 20)
// clang-format on

int main(int argc, char* argv[])
{
    primbench::settings settings;
    settings.size = 512 * primbench::MiB;
    primbench::executor executor(argc, argv, settings);

    BENCHMARK_TYPE(int32_t, 128)
    BENCHMARK_TYPE(int32_t, 256)
    BENCHMARK_TYPE(int32_t, 512)

    BENCHMARK_TYPE(uint8_t, 128)
    BENCHMARK_TYPE(uint8_t, 256)
    BENCHMARK_TYPE(uint8_t, 512)

    BENCHMARK_TYPE(int64_t, 128)
    BENCHMARK_TYPE(int64_t, 256)
    BENCHMARK_TYPE(int64_t, 512)

    BENCHMARK_TYPE(rocprim::int128_t, 128)
    BENCHMARK_TYPE(rocprim::int128_t, 256)
    BENCHMARK_TYPE(rocprim::int128_t, 512)

    BENCHMARK_TYPE(rocprim::uint128_t, 128)
    BENCHMARK_TYPE(rocprim::uint128_t, 256)
    BENCHMARK_TYPE(rocprim::uint128_t, 512)

    executor.run();
}
