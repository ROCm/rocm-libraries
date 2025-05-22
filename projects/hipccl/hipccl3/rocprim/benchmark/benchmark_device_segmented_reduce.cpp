// MIT License
//
// Copyright (c) 2017-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "benchmark_device_segmented_reduce.parallel.hpp"
#include "benchmark_utils.hpp"

#include "../common/utils_custom_type.hpp"

// HIP API
#include <hip/hip_runtime.h>

// rocPRIM
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
    executor.queue_instance(device_segmented_reduce_benchmark<T>(SEGMENTS));

#define BENCHMARK_TYPE(type)     \
    CREATE_BENCHMARK(type, 1)    \
    CREATE_BENCHMARK(type, 10)   \
    CREATE_BENCHMARK(type, 100)  \
    CREATE_BENCHMARK(type, 1000) \
    CREATE_BENCHMARK(type, 10000)

int main(int argc, char* argv[])
{
    benchmark_utils::executor executor(argc, argv, 128 * benchmark_utils::MiB, 10, 5);

#ifndef BENCHMARK_CONFIG_TUNING
    using custom_float2  = common::custom_type<float, float>;
    using custom_double2 = common::custom_type<double, double>;

    BENCHMARK_TYPE(float)
    BENCHMARK_TYPE(double)
    BENCHMARK_TYPE(int8_t)
    BENCHMARK_TYPE(uint8_t)
    BENCHMARK_TYPE(rocprim::half)
    BENCHMARK_TYPE(int)
    BENCHMARK_TYPE(custom_float2)
    BENCHMARK_TYPE(custom_double2)
    BENCHMARK_TYPE(rocprim::int128_t)
    BENCHMARK_TYPE(rocprim::uint128_t)
#endif

    executor.run();
}
