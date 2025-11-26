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

#include "benchmark_device_partial_sort_copy.hpp"
#include "primbench.hpp"

#include "../common/utils_custom_type.hpp"

#include <hip/hip_runtime.h>

#include <rocprim/types.hpp>

#include <cstddef>
#include <stdint.h>
#include <string>
#include <vector>

#define CREATE_BENCHMARK_PARTIAL_SORT_COPY(TYPE, SMALL_N) \
    executor.queue<device_partial_sort_copy_benchmark<TYPE>>(SMALL_N);

#define CREATE_BENCHMARK(TYPE)                          \
    {                                                   \
        CREATE_BENCHMARK_PARTIAL_SORT_COPY(TYPE, true)  \
        CREATE_BENCHMARK_PARTIAL_SORT_COPY(TYPE, false) \
    }

int main(int argc, char* argv[])
{
    primbench::executor executor(argc, argv, 128 * primbench::MiB, primbench::flags::sync);

    CREATE_BENCHMARK(int)
    CREATE_BENCHMARK(long long)
    CREATE_BENCHMARK(int8_t)
    CREATE_BENCHMARK(uint8_t)
    CREATE_BENCHMARK(rocprim::half)
    CREATE_BENCHMARK(short)
    CREATE_BENCHMARK(float)
    CREATE_BENCHMARK(rocprim::int128_t)
    CREATE_BENCHMARK(rocprim::uint128_t)

    CREATE_BENCHMARK(custom_float2)
    CREATE_BENCHMARK(custom_double2)
    CREATE_BENCHMARK(custom_int2)
    CREATE_BENCHMARK(custom_char_double)
    CREATE_BENCHMARK(custom_longlong_double)

    executor.run();
}
