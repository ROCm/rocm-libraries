// MIT License
//
// Copyright (c) 2017-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "benchmark_warp_sort.hpp"

#include "primbench.hpp"

#define CREATE_SORT_BENCHMARK(K, BS, WS, IPT)                  \
    if(is_warp_size_supported(WS, device_id))                  \
    {                                                          \
        executor.queue<warp_sort_benchmark<K, BS, WS, IPT>>(); \
    }

#define CREATE_SORTBYKEY_BENCHMARK(K, V, BS, WS, IPT)             \
    if(is_warp_size_supported(WS, device_id))                     \
    {                                                             \
        executor.queue<warp_sort_benchmark<K, BS, WS, IPT, V>>(); \
    }

#define BENCHMARK_TYPE(type)                \
    CREATE_SORT_BENCHMARK(type, 64, 64, 1)  \
    CREATE_SORT_BENCHMARK(type, 64, 64, 2)  \
    CREATE_SORT_BENCHMARK(type, 64, 64, 4)  \
    CREATE_SORT_BENCHMARK(type, 128, 64, 1) \
    CREATE_SORT_BENCHMARK(type, 128, 64, 2) \
    CREATE_SORT_BENCHMARK(type, 128, 64, 4) \
    CREATE_SORT_BENCHMARK(type, 256, 64, 1) \
    CREATE_SORT_BENCHMARK(type, 256, 64, 2) \
    CREATE_SORT_BENCHMARK(type, 256, 64, 4) \
    CREATE_SORT_BENCHMARK(type, 64, 32, 1)  \
    CREATE_SORT_BENCHMARK(type, 64, 32, 2)  \
    CREATE_SORT_BENCHMARK(type, 64, 16, 1)  \
    CREATE_SORT_BENCHMARK(type, 64, 16, 2)  \
    CREATE_SORT_BENCHMARK(type, 64, 16, 4)

#define BENCHMARK_KEY_TYPE(type, value)                 \
    CREATE_SORTBYKEY_BENCHMARK(type, value, 64, 64, 1)  \
    CREATE_SORTBYKEY_BENCHMARK(type, value, 64, 64, 2)  \
    CREATE_SORTBYKEY_BENCHMARK(type, value, 64, 64, 4)  \
    CREATE_SORTBYKEY_BENCHMARK(type, value, 256, 64, 1) \
    CREATE_SORTBYKEY_BENCHMARK(type, value, 256, 64, 2) \
    CREATE_SORTBYKEY_BENCHMARK(type, value, 256, 64, 4)

int main(int argc, char* argv[])
{
    primbench::settings settings;
    settings.size                    = 128 * primbench::MiB;
    settings.noise_tolerance_percent = 2;
    primbench::executor executor(argc, argv, settings);

    int device_id;
    HIP_CHECK(hipGetDevice(&device_id));

    BENCHMARK_TYPE(int32_t)
    BENCHMARK_TYPE(float)
    BENCHMARK_TYPE(double)
    BENCHMARK_TYPE(int8_t)
    BENCHMARK_TYPE(uint8_t)
    BENCHMARK_TYPE(rocprim::half)
    BENCHMARK_TYPE(rocprim::int128_t)
    BENCHMARK_TYPE(rocprim::uint128_t)

    BENCHMARK_KEY_TYPE(float, float)
    BENCHMARK_KEY_TYPE(uint32_t, int32_t)
    BENCHMARK_KEY_TYPE(int32_t, custom_f64_f64)
    BENCHMARK_KEY_TYPE(int32_t, custom_i32_f64)
    BENCHMARK_KEY_TYPE(custom_i32_i32, custom_f64_f64)
    BENCHMARK_KEY_TYPE(custom_i32_i32, custom_i8_f64)
    BENCHMARK_KEY_TYPE(custom_i32_i32, custom_i64_f64)
    BENCHMARK_KEY_TYPE(int8_t, int8_t)
    BENCHMARK_KEY_TYPE(uint8_t, uint8_t)
    BENCHMARK_KEY_TYPE(rocprim::half, rocprim::half)
    BENCHMARK_KEY_TYPE(rocprim::int128_t, rocprim::int128_t)
    BENCHMARK_KEY_TYPE(rocprim::uint128_t, rocprim::uint128_t)

    executor.run();
}
