// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "benchmark_warp_sort_stable.hpp"

#include "primbench.hpp"

#define CREATE_SORT_STABLE_BENCHMARK(K, BS, WS, IPT, ALGO)                                       \
    if(is_warp_size_supported(WS, device_id))                                                    \
    {                                                                                            \
        executor.queue<warp_sort_stable_benchmark<K, BS, WS, IPT, rocprim::empty_type, ALGO>>(); \
    }

#define CREATE_SORTBYKEY_STABLE_BENCHMARK(K, V, BS, WS, IPT, ALGO)             \
    if(is_warp_size_supported(WS, device_id))                                  \
    {                                                                          \
        executor.queue<warp_sort_stable_benchmark<K, BS, WS, IPT, V, ALGO>>(); \
    }

// clang-format off
#define BENCHMARK_TYPE_STABLE(type)                               \
    CREATE_SORT_STABLE_BENCHMARK(type, 64, 64, 1, rocprim::warp_sort_stable_algorithm::merge_path)  \
    CREATE_SORT_STABLE_BENCHMARK(type, 64, 64, 2, rocprim::warp_sort_stable_algorithm::merge_path)  \
    CREATE_SORT_STABLE_BENCHMARK(type, 64, 64, 4, rocprim::warp_sort_stable_algorithm::merge_path)  \
    CREATE_SORT_STABLE_BENCHMARK(type, 128, 64, 1, rocprim::warp_sort_stable_algorithm::merge_path) \
    CREATE_SORT_STABLE_BENCHMARK(type, 128, 64, 2, rocprim::warp_sort_stable_algorithm::merge_path) \
    CREATE_SORT_STABLE_BENCHMARK(type, 128, 64, 4, rocprim::warp_sort_stable_algorithm::merge_path) \
    CREATE_SORT_STABLE_BENCHMARK(type, 256, 64, 1, rocprim::warp_sort_stable_algorithm::merge_path) \
    CREATE_SORT_STABLE_BENCHMARK(type, 256, 64, 2, rocprim::warp_sort_stable_algorithm::merge_path) \
    CREATE_SORT_STABLE_BENCHMARK(type, 256, 64, 4, rocprim::warp_sort_stable_algorithm::merge_path) \
    CREATE_SORT_STABLE_BENCHMARK(type, 64, 32, 1, rocprim::warp_sort_stable_algorithm::merge_path)  \
    CREATE_SORT_STABLE_BENCHMARK(type, 64, 32, 2, rocprim::warp_sort_stable_algorithm::merge_path)  \
    CREATE_SORT_STABLE_BENCHMARK(type, 64, 16, 1, rocprim::warp_sort_stable_algorithm::merge_path)  \
    CREATE_SORT_STABLE_BENCHMARK(type, 64, 16, 2, rocprim::warp_sort_stable_algorithm::merge_path)  \
    CREATE_SORT_STABLE_BENCHMARK(type, 64, 16, 4, rocprim::warp_sort_stable_algorithm::merge_path)  \
    CREATE_SORT_STABLE_BENCHMARK(type, 64, 64, 1, rocprim::warp_sort_stable_algorithm::shuffle)     \
    CREATE_SORT_STABLE_BENCHMARK(type, 64, 64, 2, rocprim::warp_sort_stable_algorithm::shuffle)     \
    CREATE_SORT_STABLE_BENCHMARK(type, 64, 64, 4, rocprim::warp_sort_stable_algorithm::shuffle)     \
    CREATE_SORT_STABLE_BENCHMARK(type, 128, 64, 1, rocprim::warp_sort_stable_algorithm::shuffle)    \
    CREATE_SORT_STABLE_BENCHMARK(type, 128, 64, 2, rocprim::warp_sort_stable_algorithm::shuffle)    \
    CREATE_SORT_STABLE_BENCHMARK(type, 128, 64, 4, rocprim::warp_sort_stable_algorithm::shuffle)    \
    CREATE_SORT_STABLE_BENCHMARK(type, 256, 64, 1, rocprim::warp_sort_stable_algorithm::shuffle)    \
    CREATE_SORT_STABLE_BENCHMARK(type, 256, 64, 2, rocprim::warp_sort_stable_algorithm::shuffle)    \
    CREATE_SORT_STABLE_BENCHMARK(type, 256, 64, 4, rocprim::warp_sort_stable_algorithm::shuffle)    \
    CREATE_SORT_STABLE_BENCHMARK(type, 64, 32, 1, rocprim::warp_sort_stable_algorithm::shuffle)     \
    CREATE_SORT_STABLE_BENCHMARK(type, 64, 32, 2, rocprim::warp_sort_stable_algorithm::shuffle)     \
    CREATE_SORT_STABLE_BENCHMARK(type, 64, 16, 1, rocprim::warp_sort_stable_algorithm::shuffle)     \
    CREATE_SORT_STABLE_BENCHMARK(type, 64, 16, 2, rocprim::warp_sort_stable_algorithm::shuffle)     \
    CREATE_SORT_STABLE_BENCHMARK(type, 64, 16, 4, rocprim::warp_sort_stable_algorithm::shuffle)
// clang-format on

// clang-format off
#define BENCHMARK_KEY_TYPE_STABLE(type, value)                                  \
    CREATE_SORTBYKEY_STABLE_BENCHMARK(type, value, 64, 64, 1, rocprim::warp_sort_stable_algorithm::merge_path)  \
    CREATE_SORTBYKEY_STABLE_BENCHMARK(type, value, 64, 64, 2, rocprim::warp_sort_stable_algorithm::merge_path)  \
    CREATE_SORTBYKEY_STABLE_BENCHMARK(type, value, 64, 64, 4, rocprim::warp_sort_stable_algorithm::merge_path)  \
    CREATE_SORTBYKEY_STABLE_BENCHMARK(type, value, 256, 64, 1, rocprim::warp_sort_stable_algorithm::merge_path) \
    CREATE_SORTBYKEY_STABLE_BENCHMARK(type, value, 256, 64, 2, rocprim::warp_sort_stable_algorithm::merge_path) \
    CREATE_SORTBYKEY_STABLE_BENCHMARK(type, value, 256, 64, 4, rocprim::warp_sort_stable_algorithm::merge_path) \
    CREATE_SORTBYKEY_STABLE_BENCHMARK(type, value, 64, 64, 1, rocprim::warp_sort_stable_algorithm::shuffle)     \
    CREATE_SORTBYKEY_STABLE_BENCHMARK(type, value, 64, 64, 2, rocprim::warp_sort_stable_algorithm::shuffle)     \
    CREATE_SORTBYKEY_STABLE_BENCHMARK(type, value, 64, 64, 4, rocprim::warp_sort_stable_algorithm::shuffle)     \
    CREATE_SORTBYKEY_STABLE_BENCHMARK(type, value, 256, 64, 1, rocprim::warp_sort_stable_algorithm::shuffle)    \
    CREATE_SORTBYKEY_STABLE_BENCHMARK(type, value, 256, 64, 2, rocprim::warp_sort_stable_algorithm::shuffle)    \
    CREATE_SORTBYKEY_STABLE_BENCHMARK(type, value, 256, 64, 4, rocprim::warp_sort_stable_algorithm::shuffle)
// clang-format on

int main(int argc, char* argv[])
{
    primbench::settings settings;
    settings.size                    = 128 * primbench::MiB;
    settings.noise_tolerance_percent = 2;
    primbench::executor executor(argc, argv, settings);

    int device_id;
    HIP_CHECK(hipGetDevice(&device_id));

    BENCHMARK_TYPE_STABLE(int32_t)
    BENCHMARK_TYPE_STABLE(float)
    BENCHMARK_TYPE_STABLE(double)
    BENCHMARK_TYPE_STABLE(int8_t)
    BENCHMARK_TYPE_STABLE(uint8_t)
    BENCHMARK_TYPE_STABLE(rocprim::half)
    BENCHMARK_TYPE_STABLE(rocprim::int128_t)
    BENCHMARK_TYPE_STABLE(rocprim::uint128_t)

    BENCHMARK_KEY_TYPE_STABLE(float, float)
    BENCHMARK_KEY_TYPE_STABLE(uint32_t, int32_t)
    BENCHMARK_KEY_TYPE_STABLE(int32_t, custom_f64_f64)
    BENCHMARK_KEY_TYPE_STABLE(int32_t, custom_i32_f64)
    BENCHMARK_KEY_TYPE_STABLE(custom_i32_i32, custom_f64_f64)
    BENCHMARK_KEY_TYPE_STABLE(custom_i32_i32, custom_i8_f64)
    BENCHMARK_KEY_TYPE_STABLE(custom_i32_i32, custom_i64_f64)
    BENCHMARK_KEY_TYPE_STABLE(int8_t, int8_t)
    BENCHMARK_KEY_TYPE_STABLE(uint8_t, uint8_t)
    BENCHMARK_KEY_TYPE_STABLE(rocprim::half, rocprim::half)
    BENCHMARK_KEY_TYPE_STABLE(rocprim::int128_t, rocprim::int128_t)
    BENCHMARK_KEY_TYPE_STABLE(rocprim::uint128_t, rocprim::uint128_t)

    executor.run();
}
