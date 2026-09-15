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

#include "benchmark_device_run_length_encode.hpp"
#include "primbench.hpp"

#include "../common/utils_custom_type.hpp"

#include <hip/hip_runtime.h>

#include <rocprim/types.hpp>

#include <cstddef>
#include <stdint.h>
#include <string>
#include <vector>

#define CREATE_BENCHMARK(T)                                      \
    executor.queue<device_run_length_encode_benchmark<T, 10>>(); \
    executor.queue<device_run_length_encode_benchmark<T, 1000>>();

int main(int argc, char* argv[])
{
    primbench::settings settings;
    settings.size = 2 * primbench::GiB;
    primbench::executor executor(argc, argv, settings);

#ifndef BENCHMARK_CONFIG_TUNING
    // Tuned types
    CREATE_BENCHMARK(rocprim::int128_t)
    CREATE_BENCHMARK(int64_t)
    CREATE_BENCHMARK(int32_t)
    CREATE_BENCHMARK(int16_t)
    CREATE_BENCHMARK(int8_t)
    CREATE_BENCHMARK(double)
    CREATE_BENCHMARK(float)
    CREATE_BENCHMARK(rocprim::half)

    #ifndef BENCHMARK_AUTOTUNED_TYPES_ONLY
    // Not tuned types
    CREATE_BENCHMARK(rocprim::uint128_t)

    // Not tuned custom types
    CREATE_BENCHMARK(custom_f32_f32)
    CREATE_BENCHMARK(custom_f64_f64)
    #endif
#endif

    executor.run();
}
