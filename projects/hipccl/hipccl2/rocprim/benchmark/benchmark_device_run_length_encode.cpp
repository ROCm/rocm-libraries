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

#include "benchmark_device_run_length_encode.parallel.hpp"
#include "benchmark_utils.hpp"

#include "../common/utils_custom_type.hpp"

// HIP API
#include <hip/hip_runtime.h>

// rocPRIM
#include <rocprim/types.hpp>

#include <cstddef>
#include <stdint.h>
#include <string>
#include <vector>

#define CREATE_ENCODE_BENCHMARK(T, ML) \
    executor.queue_instance(device_run_length_encode_benchmark<T, ML>());

template<size_t MaxLength>
void add_encode_benchmarks(benchmark_utils::executor& executor)
{
    using custom_float2  = common::custom_type<float, float>;
    using custom_double2 = common::custom_type<double, double>;

    // all tuned types
    CREATE_ENCODE_BENCHMARK(int8_t, MaxLength)
    CREATE_ENCODE_BENCHMARK(int16_t, MaxLength)
    CREATE_ENCODE_BENCHMARK(int32_t, MaxLength)
    CREATE_ENCODE_BENCHMARK(int64_t, MaxLength)
    CREATE_ENCODE_BENCHMARK(rocprim::int128_t, MaxLength)
    CREATE_ENCODE_BENCHMARK(rocprim::uint128_t, MaxLength)
    CREATE_ENCODE_BENCHMARK(rocprim::half, MaxLength)
    CREATE_ENCODE_BENCHMARK(float, MaxLength)
    CREATE_ENCODE_BENCHMARK(double, MaxLength)
    // custom types
    CREATE_ENCODE_BENCHMARK(custom_float2, MaxLength)
    CREATE_ENCODE_BENCHMARK(custom_double2, MaxLength)
}

int main(int argc, char* argv[])
{
    benchmark_utils::executor executor(argc, argv, 2 * benchmark_utils::GiB, 10, 10);

#ifndef BENCHMARK_CONFIG_TUNING

    add_encode_benchmarks<1000>(executor);
    add_encode_benchmarks<10>(executor);

#endif

    executor.run();
}
