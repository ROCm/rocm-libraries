// MIT License
//
// Copyright (c) 2019-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "benchmark_device_binary_search.parallel.hpp"

#include "benchmark_utils.hpp"

#include "../common/utils_custom_type.hpp"
#include "../common/utils_device_ptr.hpp"

// HIP API
#include <hip/hip_runtime.h>

// rocPRIM
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
    executor.queue_instance(                                               \
        device_binary_search_benchmark<SUBALGORITHM, VALUE_TYPE, OUTPUT_TYPE, K, SORTED>());

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
    BENCHMARK_TYPE_TUNING(VALUE_TYPE, int)               \
    BENCHMARK_TYPE_TUNING(VALUE_TYPE, short)             \
    BENCHMARK_TYPE_TUNING(VALUE_TYPE, int8_t)

int main(int argc, char* argv[])
{
    benchmark_utils::executor executor(argc, argv, 128 * benchmark_utils::MiB, 10, 5);

#ifndef BENCHMARK_CONFIG_TUNING
    // Tuned types
    BENCHMARK_TYPES_TUNING(rocprim::int128_t)
    BENCHMARK_TYPES_TUNING(int64_t)
    BENCHMARK_TYPES_TUNING(int)
    BENCHMARK_TYPES_TUNING(short)
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
    using custom_float2  = common::custom_type<float, float>;
    using custom_double2 = common::custom_type<double, double>;

    BENCHMARK_TYPE(custom_float2)
    BENCHMARK_TYPE(custom_double2)
    #endif
#endif

    executor.run();
}
