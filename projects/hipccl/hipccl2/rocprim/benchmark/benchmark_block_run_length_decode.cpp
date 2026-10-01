// MIT License
//
// Copyright (c) 2021-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "benchmark_block_run_length_decode.hpp"

#include "primbench.hpp"

#define CREATE_BENCHMARK(IT, OT, MINRL, MAXRL, BS, RPT, DIPT) \
    executor.queue<block_run_length_decode_benchmark<IT, OT, MINRL, MAXRL, BS, RPT, DIPT>>();

int main(int argc, char* argv[])
{
    primbench::settings settings;
    settings.size                    = 128 * primbench::MiB;
    settings.min_gpu_ms_per_batch    = 100;
    settings.noise_tolerance_percent = 2;
    primbench::executor executor(argc, argv, settings);

    CREATE_BENCHMARK(int32_t, int32_t, 1, 5, 128, 2, 4)
    CREATE_BENCHMARK(int32_t, int32_t, 1, 10, 128, 2, 4)
    CREATE_BENCHMARK(int32_t, int32_t, 1, 50, 128, 2, 4)
    CREATE_BENCHMARK(int32_t, int32_t, 1, 100, 128, 2, 4)
    CREATE_BENCHMARK(int32_t, int32_t, 1, 500, 128, 2, 4)
    CREATE_BENCHMARK(int32_t, int32_t, 1, 1000, 128, 2, 4)
    CREATE_BENCHMARK(int32_t, int32_t, 1, 5000, 128, 2, 4)

    CREATE_BENCHMARK(double, int64_t, 1, 5, 128, 2, 4)
    CREATE_BENCHMARK(double, int64_t, 1, 10, 128, 2, 4)
    CREATE_BENCHMARK(double, int64_t, 1, 50, 128, 2, 4)
    CREATE_BENCHMARK(double, int64_t, 1, 100, 128, 2, 4)
    CREATE_BENCHMARK(double, int64_t, 1, 500, 128, 2, 4)
    CREATE_BENCHMARK(double, int64_t, 1, 1000, 128, 2, 4)
    CREATE_BENCHMARK(double, int64_t, 1, 5000, 128, 2, 4)

    CREATE_BENCHMARK(rocprim::int128_t, rocprim::int128_t, 1, 5, 128, 2, 4)
    CREATE_BENCHMARK(rocprim::int128_t, rocprim::int128_t, 1, 10, 128, 2, 4)
    CREATE_BENCHMARK(rocprim::int128_t, rocprim::int128_t, 1, 50, 128, 2, 4)
    CREATE_BENCHMARK(rocprim::int128_t, rocprim::int128_t, 1, 100, 128, 2, 4)
    CREATE_BENCHMARK(rocprim::int128_t, rocprim::int128_t, 1, 500, 128, 2, 4)
    CREATE_BENCHMARK(rocprim::int128_t, rocprim::int128_t, 1, 1000, 128, 2, 4)
    CREATE_BENCHMARK(rocprim::int128_t, rocprim::int128_t, 1, 5000, 128, 2, 4)

    CREATE_BENCHMARK(rocprim::uint128_t, rocprim::uint128_t, 1, 5, 128, 2, 4)
    CREATE_BENCHMARK(rocprim::uint128_t, rocprim::uint128_t, 1, 10, 128, 2, 4)
    CREATE_BENCHMARK(rocprim::uint128_t, rocprim::uint128_t, 1, 50, 128, 2, 4)
    CREATE_BENCHMARK(rocprim::uint128_t, rocprim::uint128_t, 1, 100, 128, 2, 4)
    CREATE_BENCHMARK(rocprim::uint128_t, rocprim::uint128_t, 1, 500, 128, 2, 4)
    CREATE_BENCHMARK(rocprim::uint128_t, rocprim::uint128_t, 1, 1000, 128, 2, 4)
    CREATE_BENCHMARK(rocprim::uint128_t, rocprim::uint128_t, 1, 5000, 128, 2, 4)

    executor.run();
}
