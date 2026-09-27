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

#include "benchmark_device_segmented_radix_sort_keys.hpp"
#include "primbench.hpp"

#include "../common/utils_data_generation.hpp"

#include <hip/hip_runtime.h>

#include <rocprim/device/device_segmented_radix_sort.hpp>
#include <rocprim/types.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <random>
#include <string>
#include <vector>
#ifndef BENCHMARK_CONFIG_TUNING
    #include <stdint.h>
#endif

// This benchmark only handles the rocprim::segmented_radix_sort_keys function. The benchmark was separated into two (keys and pairs),
// because the binary became too large to link. Runs into a "relocation R_X86_64_PC32 out of range" error.
// This happens partially, because of the algorithm has 4 kernels, and decides at runtime which one to call.

template<typename KeyT>
void add_benchmarks(primbench::executor& executor, size_t bytes)
{
    constexpr std::array<size_t, 8> segment_counts{10, 100, 1000, 2500, 5000, 7500, 10000, 100000};
    constexpr std::array<size_t, 4> segment_lengths{30, 256, 3000, 300000};

    constexpr size_t min_size = 30000;
    size_t           max_size = bytes / sizeof(KeyT);

    for(const auto segment_count : segment_counts)
    {
        for(const auto segment_length : segment_lengths)
        {
            const auto number_of_elements = segment_count * segment_length;
            if(number_of_elements < min_size || number_of_elements > max_size)
            {
                continue;
            }

            executor.queue<device_segmented_radix_sort_keys_benchmark<KeyT>>(segment_count,
                                                                             segment_length);
        }
    }
}

int main(int argc, char* argv[])
{
    size_t bytes = 128 * primbench::MiB;

    primbench::settings settings;
    settings.size = bytes;
    primbench::executor executor(argc, argv, settings, primbench::flags::sync);

#ifndef BENCHMARK_CONFIG_TUNING
    // Tuned types
    add_benchmarks<rocprim::int128_t>(executor, bytes);
    add_benchmarks<int64_t>(executor, bytes);
    add_benchmarks<int32_t>(executor, bytes);
    add_benchmarks<int16_t>(executor, bytes);
    add_benchmarks<int8_t>(executor, bytes);
    add_benchmarks<double>(executor, bytes);
    add_benchmarks<float>(executor, bytes);
    add_benchmarks<rocprim::half>(executor, bytes);

    #ifndef BENCHMARK_AUTOTUNED_TYPES_ONLY
    // Not tuned types
    add_benchmarks<uint8_t>(executor, bytes);
    add_benchmarks<rocprim::uint128_t>(executor, bytes);
    #endif
#endif

    executor.run();
}
