// MIT License
//
// Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "benchmark_device_topk_air_topk.hpp"
#include "benchmark_utils.hpp"

// HIP API
#include <hip/hip_runtime.h>

#include <cstddef>
#include <string>
#include <vector>

#define CREATE_TOPK_AIR_TOPK_BENCHMARK(SMALL_K, ADVERSARIAL_DISTRIBUTION, ...) \
    executor.queue_instance(                                                   \
        device_air_topk_benchmark<__VA_ARGS__>(SMALL_K, ADVERSARIAL_DISTRIBUTION));

#define CREATE_BENCHMARK(...)                                \
    CREATE_TOPK_AIR_TOPK_BENCHMARK(true, true, __VA_ARGS__)  \
    CREATE_TOPK_AIR_TOPK_BENCHMARK(true, false, __VA_ARGS__) \
    CREATE_TOPK_AIR_TOPK_BENCHMARK(false, true, __VA_ARGS__) \
    CREATE_TOPK_AIR_TOPK_BENCHMARK(false, false, __VA_ARGS__)

int main(int argc, char* argv[])
{
    benchmark_utils::executor executor(argc, argv, 128 * benchmark_utils::MiB, 10, 5);
    using custom_float2  = common::custom_type<float, float>;
    using custom_double2 = common::custom_type<double, double>;
    // Note: custom keys not supported yet
    // using custom_key_float_int16 = common::custom_type<float, int16_t>;
    // using custom_key_int2            = common::custom_type<int, int>;
    // using custom_key_char_double     = common::custom_type<char, double>;
    // using custom_key_longlong_double = common::custom_type<long long, double>;

    // Fundamental key benchmarks, comparable with nth element and radix sort
    //  Integer
    CREATE_BENCHMARK(int8_t)
    CREATE_BENCHMARK(uint8_t)
    CREATE_BENCHMARK(short)
    CREATE_BENCHMARK(int)
    CREATE_BENCHMARK(long long)
    CREATE_BENCHMARK(rocprim::int128_t)
    CREATE_BENCHMARK(rocprim::uint128_t)
    //  Float
    CREATE_BENCHMARK(rocprim::half)
    CREATE_BENCHMARK(float)
    CREATE_BENCHMARK(double)

    // Custom key benchmarks (not supported yet)
    //  Comparable with nth element
    // CREATE_BENCHMARK(custom_float2)
    // CREATE_BENCHMARK(custom_double2)
    // CREATE_BENCHMARK(custom_key_int2)
    // CREATE_BENCHMARK(custom_key_char_double)
    // CREATE_BENCHMARK(custom_key_longlong_double)
    //  Comparable with radix sort
    // CREATE_BENCHMARK(custom_key_float_int16)

    // Pair benchmarks, comparable with radix sort
    //  With fundamental key
    CREATE_BENCHMARK(int, float)
    CREATE_BENCHMARK(int, double)
    CREATE_BENCHMARK(int, float2)
    CREATE_BENCHMARK(int, custom_float2)
    CREATE_BENCHMARK(int, double2)
    CREATE_BENCHMARK(int, custom_double2)
    CREATE_BENCHMARK(long long, float)
    CREATE_BENCHMARK(long long, double)
    CREATE_BENCHMARK(long long, float2)
    CREATE_BENCHMARK(long long, custom_float2)
    CREATE_BENCHMARK(long long, double2)
    CREATE_BENCHMARK(long long, custom_double2)
    CREATE_BENCHMARK(int8_t, int8_t)
    CREATE_BENCHMARK(uint8_t, uint8_t)
    CREATE_BENCHMARK(rocprim::half, rocprim::half)
    CREATE_BENCHMARK(rocprim::int128_t, rocprim::int128_t)
    CREATE_BENCHMARK(rocprim::uint128_t, rocprim::uint128_t)

    executor.run();
}
