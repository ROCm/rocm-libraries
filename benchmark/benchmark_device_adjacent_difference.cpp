// MIT License
//
// Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "benchmark_device_adjacent_difference.parallel.hpp"

#ifndef BENCHMARK_CONFIG_TUNING
    #include "../common/device_adjacent_difference.hpp"
    #include "../common/utils_custom_type.hpp"
#endif

// HIP API
#include <hip/hip_runtime_api.h>

// rocPRIM
#ifndef BENCHMARK_CONFIG_TUNING
    #include <rocprim/types.hpp>
#endif

#include <cstddef>
#include <string>
#include <vector>
#ifndef BENCHMARK_CONFIG_TUNING
    #include <stdint.h>
#endif

#define CREATE_BENCHMARK(T, Left, Aliasing) \
    executor.queue_instance(device_adjacent_difference_benchmark<T, Left, Aliasing>());

// clang-format off
#define CREATE_BENCHMARKS(T) \
    CREATE_BENCHMARK(T, true,  common::api_variant::no_alias) \
    CREATE_BENCHMARK(T, true,  common::api_variant::in_place) \
    CREATE_BENCHMARK(T, false, common::api_variant::no_alias) \
    CREATE_BENCHMARK(T, false, common::api_variant::in_place)
// clang-format on

int main(int argc, char* argv[])
{
    benchmark_utils::executor executor(argc, argv, 2 * benchmark_utils::GiB, 10, 5);

#ifndef BENCHMARK_CONFIG_TUNING
    using custom_float2  = common::custom_type<float, float>;
    using custom_double2 = common::custom_type<double, double>;

    CREATE_BENCHMARKS(int);
    CREATE_BENCHMARKS(std::int64_t);

    CREATE_BENCHMARKS(uint8_t);
    CREATE_BENCHMARKS(rocprim::half);

    CREATE_BENCHMARKS(float);
    CREATE_BENCHMARKS(double);

    CREATE_BENCHMARKS(custom_float2);
    CREATE_BENCHMARKS(custom_double2);

    CREATE_BENCHMARKS(rocprim::int128_t);
    CREATE_BENCHMARKS(rocprim::uint128_t);
#endif

    executor.run();
}
