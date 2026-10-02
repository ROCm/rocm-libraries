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

#include "benchmark_device_find_first_of.hpp"

#include "primbench.hpp"

#define CREATE_BENCHMARK_FIND_FIRST_OF(TYPE, KEYS_SIZE, FIRST_OCCURENCE) \
    executor.queue<device_find_first_of_benchmark<TYPE>>(KEYS_SIZE, FIRST_OCCURENCE);

// clang-format off
#define CREATE_BENCHMARK0(TYPE, KEYS_SIZE) \
    CREATE_BENCHMARK_FIND_FIRST_OF(TYPE, KEYS_SIZE, 0.1) \
    CREATE_BENCHMARK_FIND_FIRST_OF(TYPE, KEYS_SIZE, 0.5) \
    CREATE_BENCHMARK_FIND_FIRST_OF(TYPE, KEYS_SIZE, 1.0)

#define CREATE_BENCHMARK(TYPE) \
        CREATE_BENCHMARK0(TYPE, 1) \
        CREATE_BENCHMARK0(TYPE, 10) \
        CREATE_BENCHMARK0(TYPE, 100) \
        CREATE_BENCHMARK0(TYPE, 1000) \
        CREATE_BENCHMARK0(TYPE, 10000)
// clang-format on

int main(int argc, char* argv[])
{
    primbench::settings settings;
    settings.size                    = 128 * primbench::MiB;
    settings.min_gpu_ms_per_batch    = 1000;
    settings.batch_window_size       = 3;
    settings.noise_tolerance_percent = 4;
    primbench::executor executor(argc, argv, settings);

#ifndef BENCHMARK_CONFIG_TUNING
    // Tuned types
    CREATE_BENCHMARK(rocprim::int128_t)
    CREATE_BENCHMARK(int64_t)
    CREATE_BENCHMARK(int32_t)
    CREATE_BENCHMARK(int16_t)
    CREATE_BENCHMARK(int8_t)

    #ifndef BENCHMARK_AUTOTUNED_TYPES_ONLY
    // Not tuned types
    CREATE_BENCHMARK(float)
    CREATE_BENCHMARK(double)
    CREATE_BENCHMARK(rocprim::uint128_t)

    // Not tuned custom types
    CREATE_BENCHMARK(custom_i32_i32)
    CREATE_BENCHMARK(custom_i64_f64)
    #endif
#endif

    executor.run();
}
