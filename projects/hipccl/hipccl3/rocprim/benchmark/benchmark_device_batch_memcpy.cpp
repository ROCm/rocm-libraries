// MIT License
//
// Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "benchmark_device_batch_memcpy.hpp"

#include "primbench.hpp"

#define CREATE_NAIVE_BENCHMARK(item_size, item_alignment, size_type, num_tlev, num_wlev, num_blev) \
    executor.queue<device_batch_memcpy_benchmark<item_size,                                        \
                                                 item_alignment,                                   \
                                                 size_type,                                        \
                                                 true,                                             \
                                                 num_tlev,                                         \
                                                 num_wlev,                                         \
                                                 num_blev>>("naive_memcpy");

#define CREATE_BENCHMARK(item_size,                              \
                         item_alignment,                         \
                         size_type,                              \
                         num_tlev,                               \
                         num_wlev,                               \
                         num_blev,                               \
                         is_memcpy,                              \
                         subalgo)                                \
    executor.queue<device_batch_memcpy_benchmark<item_size,      \
                                                 item_alignment, \
                                                 size_type,      \
                                                 is_memcpy,      \
                                                 num_tlev,       \
                                                 num_wlev,       \
                                                 num_blev>>(subalgo);

#define CREATE_NORMAL_BENCHMARK(item_size,      \
                                item_alignment, \
                                size_type,      \
                                num_tlev,       \
                                num_wlev,       \
                                num_blev)       \
    CREATE_BENCHMARK(item_size,                 \
                     item_alignment,            \
                     size_type,                 \
                     num_tlev,                  \
                     num_wlev,                  \
                     num_blev,                  \
                     true,                      \
                     "batch_memcpy")            \
    CREATE_BENCHMARK(item_size,                 \
                     item_alignment,            \
                     size_type,                 \
                     num_tlev,                  \
                     num_wlev,                  \
                     num_blev,                  \
                     false,                     \
                     "batch_copy")

#ifndef BUILD_NAIVE_BENCHMARK
    #define BENCHMARK_TYPE(item_size, item_alignment)                                        \
        CREATE_NORMAL_BENCHMARK(item_size, item_alignment, uint32_t, 100000, 0, 0)           \
        CREATE_NORMAL_BENCHMARK(item_size, item_alignment, uint32_t, 0, 100000, 0)           \
        CREATE_NORMAL_BENCHMARK(item_size, item_alignment, uint32_t, 0, 0, 1000)             \
        CREATE_NORMAL_BENCHMARK(item_size, item_alignment, uint32_t, 1000, 1000, 1000)       \
        CREATE_NORMAL_BENCHMARK(item_size, item_alignment, rocprim::uint128_t, 100000, 0, 0) \
        CREATE_NORMAL_BENCHMARK(item_size, item_alignment, rocprim::uint128_t, 0, 100000, 0) \
        CREATE_NORMAL_BENCHMARK(item_size, item_alignment, rocprim::uint128_t, 0, 0, 1000)   \
        CREATE_NORMAL_BENCHMARK(item_size, item_alignment, rocprim::uint128_t, 1000, 1000, 1000)
#else
    #define BENCHMARK_TYPE(item_size, item_alignment)                                            \
        CREATE_NORMAL_BENCHMARK(item_size, item_alignment, uint32_t, 100000, 0, 0)               \
        CREATE_NORMAL_BENCHMARK(item_size, item_alignment, uint32_t, 0, 100000, 0)               \
        CREATE_NORMAL_BENCHMARK(item_size, item_alignment, uint32_t, 0, 0, 1000)                 \
        CREATE_NORMAL_BENCHMARK(item_size, item_alignment, uint32_t, 1000, 1000, 1000)           \
        CREATE_NAIVE_BENCHMARK(item_size, item_alignment, uint32_t, 100000, 0, 0)                \
        CREATE_NAIVE_BENCHMARK(item_size, item_alignment, uint32_t, 0, 100000, 0)                \
        CREATE_NAIVE_BENCHMARK(item_size, item_alignment, uint32_t, 0, 0, 1000)                  \
        CREATE_NAIVE_BENCHMARK(item_size, item_alignment, uint32_t, 1000, 1000, 1000)            \
        CREATE_NORMAL_BENCHMARK(item_size, item_alignment, rocprim::uint128_t, 100000, 0, 0)     \
        CREATE_NORMAL_BENCHMARK(item_size, item_alignment, rocprim::uint128_t, 0, 100000, 0)     \
        CREATE_NORMAL_BENCHMARK(item_size, item_alignment, rocprim::uint128_t, 0, 0, 1000)       \
        CREATE_NORMAL_BENCHMARK(item_size, item_alignment, rocprim::uint128_t, 1000, 1000, 1000) \
        CREATE_NAIVE_BENCHMARK(item_size, item_alignment, rocprim::uint128_t, 100000, 0, 0)      \
        CREATE_NAIVE_BENCHMARK(item_size, item_alignment, rocprim::uint128_t, 0, 100000, 0)      \
        CREATE_NAIVE_BENCHMARK(item_size, item_alignment, rocprim::uint128_t, 0, 0, 1000)        \
        CREATE_NAIVE_BENCHMARK(item_size, item_alignment, rocprim::uint128_t, 1000, 1000, 1000)
#endif //BUILD_NAIVE_BENCHMARK

int main(int argc, char* argv[])
{
    // Set the number of bytes to 1, because prepare_data() later on calculates it.
    primbench::settings settings;
    settings.size = 1;
    primbench::executor executor(argc, argv, settings);

    BENCHMARK_TYPE(1, 1)
    BENCHMARK_TYPE(1, 2)
    BENCHMARK_TYPE(1, 4)
    BENCHMARK_TYPE(1, 8)
    BENCHMARK_TYPE(2, 2)
    BENCHMARK_TYPE(4, 4)
    BENCHMARK_TYPE(8, 8)

    executor.run();
}
