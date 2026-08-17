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

#include "benchmark_device_segmented_reduce.hpp"
#include "primbench.hpp"

#include "../common/utils_custom_type.hpp"

#include <hip/hip_runtime.h>

#include <rocprim/device/device_segmented_reduce.hpp>
#include <rocprim/functional.hpp>
#include <rocprim/types.hpp>

#include <cmath>
#include <cstddef>
#include <numeric>
#include <random>
#include <stdint.h>
#include <string>
#include <vector>

#define CREATE_BENCHMARK(T, SEGMENTS) \
    executor.queue<device_segmented_reduce_benchmark<T>>(SEGMENTS);

#define BENCHMARK_TYPE(type)     \
    CREATE_BENCHMARK(type, 1)    \
    CREATE_BENCHMARK(type, 10)   \
    CREATE_BENCHMARK(type, 100)  \
    CREATE_BENCHMARK(type, 1000) \
    CREATE_BENCHMARK(type, 10000)

int main(int argc, char* argv[])
{
    primbench::settings settings;
    settings.size                 = 128 * primbench::MiB;
    settings.min_gpu_ms_per_batch = 100;
    primbench::executor executor(argc, argv, settings);

#ifndef BENCHMARK_CONFIG_TUNING
    // Tuned types
    BENCHMARK_TYPE(rocprim::int128_t)
    BENCHMARK_TYPE(int64_t)
    BENCHMARK_TYPE(int32_t)
    BENCHMARK_TYPE(int16_t)
    BENCHMARK_TYPE(int8_t)
    BENCHMARK_TYPE(double)
    BENCHMARK_TYPE(float)
    BENCHMARK_TYPE(rocprim::half)

    #ifndef BENCHMARK_AUTOTUNED_TYPES_ONLY
    // Not tuned types
    BENCHMARK_TYPE(uint8_t)
    BENCHMARK_TYPE(rocprim::uint128_t)

    // Not tuned custom types
    BENCHMARK_TYPE(custom_f32_f32)
    BENCHMARK_TYPE(custom_f64_f64)
    #endif
#endif

    executor.run();
}
