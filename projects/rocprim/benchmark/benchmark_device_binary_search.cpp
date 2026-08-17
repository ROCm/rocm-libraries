// MIT License
//
// Copyright (c) 2019-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "benchmark_device_binary_search.hpp"

#include "primbench.hpp"

#include "../common/utils_custom_type.hpp"
#include "../common/utils_device_ptr.hpp"

#include <hip/hip_runtime.h>

#include <rocprim/device/config_types.hpp>
#include <rocprim/functional.hpp>
#include <rocprim/types.hpp>

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <string>
#include <type_traits>
#include <vector>
#ifndef BENCHMARK_CONFIG_TUNING
    #include <stdint.h>
#endif

#define CREATE_BENCHMARK(VALUE_TYPE, OUTPUT_TYPE, K, SORTED, SUBALGORITHM) \
    executor.queue<                                                        \
        device_binary_search_benchmark<SUBALGORITHM, VALUE_TYPE, OUTPUT_TYPE, K, SORTED>>();

#define BENCHMARK_ALGORITHMS(VALUE_TYPE, OUTPUT_TYPE, K, SORTED)                     \
    CREATE_BENCHMARK(VALUE_TYPE, OUTPUT_TYPE, K, SORTED, binary_search_subalgorithm) \
    CREATE_BENCHMARK(VALUE_TYPE, OUTPUT_TYPE, K, SORTED, lower_bound_subalgorithm)   \
    CREATE_BENCHMARK(VALUE_TYPE, OUTPUT_TYPE, K, SORTED, upper_bound_subalgorithm)

#define BENCHMARK_TYPE(VALUE_TYPE)                     \
    BENCHMARK_ALGORITHMS(VALUE_TYPE, size_t, 10, true) \
    BENCHMARK_ALGORITHMS(VALUE_TYPE, size_t, 10, false)

#define BENCHMARK_TYPE_TUNING(VALUE_TYPE, OUTPUT_TYPE)      \
    BENCHMARK_ALGORITHMS(VALUE_TYPE, OUTPUT_TYPE, 10, true) \
    BENCHMARK_ALGORITHMS(VALUE_TYPE, OUTPUT_TYPE, 10, false)

// All of the limited tuned types
#define BENCHMARK_TYPES_TUNING(VALUE_TYPE)               \
    BENCHMARK_TYPE_TUNING(VALUE_TYPE, rocprim::int128_t) \
    BENCHMARK_TYPE_TUNING(VALUE_TYPE, int64_t)           \
    BENCHMARK_TYPE_TUNING(VALUE_TYPE, int32_t)           \
    BENCHMARK_TYPE_TUNING(VALUE_TYPE, int16_t)           \
    BENCHMARK_TYPE_TUNING(VALUE_TYPE, int8_t)

int main(int argc, char* argv[])
{
    primbench::settings settings;
    settings.size = 128 * primbench::MiB;
    primbench::executor executor(argc, argv, settings);

#ifndef BENCHMARK_CONFIG_TUNING
    // Tuned types
    BENCHMARK_TYPES_TUNING(rocprim::int128_t)
    BENCHMARK_TYPES_TUNING(int64_t)
    BENCHMARK_TYPES_TUNING(int32_t)
    BENCHMARK_TYPES_TUNING(int16_t)
    BENCHMARK_TYPES_TUNING(int8_t)
    BENCHMARK_TYPES_TUNING(double)
    BENCHMARK_TYPES_TUNING(float)
    BENCHMARK_TYPES_TUNING(rocprim::half)

    #ifndef BENCHMARK_AUTOTUNED_TYPES_ONLY
    // Not tuned types
    BENCHMARK_TYPE(float)
    BENCHMARK_TYPE(double)
    BENCHMARK_TYPE(uint8_t)
    BENCHMARK_TYPE(rocprim::half)
    BENCHMARK_TYPE(rocprim::uint128_t)

    // Not tuned custom types
    BENCHMARK_TYPE(custom_f32_f32)
    BENCHMARK_TYPE(custom_f64_f64)
    #endif
#endif

    executor.run();
}
