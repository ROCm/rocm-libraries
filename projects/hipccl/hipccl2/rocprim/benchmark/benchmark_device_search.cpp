// MIT License
//
// Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "benchmark_device_search.hpp"
#include "primbench.hpp"

#include "../common/utils_custom_type.hpp"

#include <rocprim/types.hpp>

#include <stdint.h>

#define CREATE_BENCHMARK_SEARCH(TYPE, KEY_SIZE, REPEATING) \
    executor.queue<device_search_benchmark<TYPE>>(KEY_SIZE, REPEATING);

#define CREATE_BENCHMARK_PATTERN(TYPE, REPEATING)                                               \
    {CREATE_BENCHMARK_SEARCH(TYPE, 10, REPEATING) CREATE_BENCHMARK_SEARCH(TYPE, 100, REPEATING) \
         CREATE_BENCHMARK_SEARCH(TYPE, 1000, REPEATING)                                         \
             CREATE_BENCHMARK_SEARCH(TYPE, 10000, REPEATING)}

#define CREATE_BENCHMARK(TYPE) \
    {CREATE_BENCHMARK_PATTERN(TYPE, true) CREATE_BENCHMARK_PATTERN(TYPE, false)}

int main(int argc, char* argv[])
{
    primbench::settings settings;
    settings.size                 = 128 * primbench::MiB;
    settings.min_gpu_ms_per_batch = 100;
    primbench::executor executor(argc, argv, settings);

    CREATE_BENCHMARK(int32_t)
    CREATE_BENCHMARK(int64_t)
    CREATE_BENCHMARK(int8_t)
    CREATE_BENCHMARK(uint8_t)
    CREATE_BENCHMARK(rocprim::half)
    CREATE_BENCHMARK(int16_t)
    CREATE_BENCHMARK(float)
    CREATE_BENCHMARK(rocprim::int128_t)
    CREATE_BENCHMARK(rocprim::uint128_t)

    CREATE_BENCHMARK(custom_f32_f32)
    CREATE_BENCHMARK(custom_f64_f64)
    CREATE_BENCHMARK(custom_i32_i32)
    CREATE_BENCHMARK(custom_i8_f64)
    CREATE_BENCHMARK(custom_i64_f64)

    executor.run();
}
