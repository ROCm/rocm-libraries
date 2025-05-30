// MIT License
//
// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "benchmark_device_find_first_of.parallel.hpp"
#include "benchmark_utils.hpp"

#ifndef BENCHMARK_CONFIG_TUNING
    #include "../common/utils_custom_type.hpp"
#endif

// HIP API
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

#define CREATE_BENCHMARK_FIND_FIRST_OF(TYPE, KEYS_SIZE, FIRST_OCCURENCE) \
    executor.queue_instance(device_find_first_of_benchmark<TYPE>(KEYS_SIZE, FIRST_OCCURENCE));

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
    benchmark_utils::executor executor(argc, argv, 128 * benchmark_utils::MiB, 10, 2);

#ifndef BENCHMARK_CONFIG_TUNING
    CREATE_BENCHMARK(int8_t)
    CREATE_BENCHMARK(int16_t)
    CREATE_BENCHMARK(int32_t)
    CREATE_BENCHMARK(float)
    CREATE_BENCHMARK(int64_t)
    CREATE_BENCHMARK(double)
    CREATE_BENCHMARK(rocprim::int128_t)
    CREATE_BENCHMARK(rocprim::uint128_t)

    using custom_int2            = common::custom_type<int, int>;
    using custom_longlong_double = common::custom_type<long long, double>;

    CREATE_BENCHMARK(custom_int2)
    CREATE_BENCHMARK(custom_longlong_double)
#endif

    executor.run();
}
