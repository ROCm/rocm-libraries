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

#include "benchmark_device_merge_sort_block_merge.hpp"
#include "primbench.hpp"

#ifndef BENCHMARK_CONFIG_TUNING
    #include "../common/utils_custom_type.hpp"
#endif

#include <hip/hip_runtime.h>

#ifndef BENCHMARK_CONFIG_TUNING
    #include <rocprim/types.hpp>
#endif

#include <cstddef>
#include <string>
#include <vector>
#ifndef BENCHMARK_CONFIG_TUNING
    #include <stdint.h>
#endif

#define CREATE_BENCHMARK(...) \
    executor.queue<device_merge_sort_block_merge_benchmark<__VA_ARGS__>>();

#define CREATE_BENCHMARK_TYPE_TUNING(KeyType)      \
    CREATE_BENCHMARK(KeyType, rocprim::empty_type) \
    CREATE_BENCHMARK(KeyType, rocprim::int128_t)   \
    CREATE_BENCHMARK(KeyType, int64_t)             \
    CREATE_BENCHMARK(KeyType, int32_t)             \
    CREATE_BENCHMARK(KeyType, int16_t)             \
    CREATE_BENCHMARK(KeyType, int8_t)

int main(int argc, char* argv[])
{
    primbench::settings settings;
    settings.size                 = 128 * primbench::MiB;
    settings.min_gpu_ms_per_batch = 100;
    primbench::executor executor(argc, argv, settings);

#ifndef BENCHMARK_CONFIG_TUNING
    // Tuned types
    CREATE_BENCHMARK_TYPE_TUNING(rocprim::int128_t)
    CREATE_BENCHMARK_TYPE_TUNING(int64_t)
    CREATE_BENCHMARK_TYPE_TUNING(int32_t)
    CREATE_BENCHMARK_TYPE_TUNING(int16_t)
    CREATE_BENCHMARK_TYPE_TUNING(int8_t)
    CREATE_BENCHMARK_TYPE_TUNING(double)
    CREATE_BENCHMARK_TYPE_TUNING(float)
    CREATE_BENCHMARK_TYPE_TUNING(rocprim::half)

    #ifndef BENCHMARK_AUTOTUNED_TYPES_ONLY
    // Not tuned types
    CREATE_BENCHMARK(uint8_t)
    CREATE_BENCHMARK(rocprim::uint128_t)

    // Not tuned custom types
    CREATE_BENCHMARK(int32_t, custom_f32_f32)
    CREATE_BENCHMARK(int64_t, custom_f64_f64)
    CREATE_BENCHMARK(custom_f64_f64, custom_f64_f64)
    CREATE_BENCHMARK(custom_i32_i32, custom_f64_f64)
    CREATE_BENCHMARK(custom_i32_i32, custom_i8_f64)
    CREATE_BENCHMARK(custom_i32_i32, custom_i64_f64)
    #endif
#endif

    executor.run();
}
