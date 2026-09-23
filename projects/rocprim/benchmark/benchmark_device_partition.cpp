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

#include "benchmark_device_partition.hpp"
#include "primbench.hpp"

#ifndef BENCHMARK_CONFIG_TUNING
    #include "../common/utils_custom_type.hpp"
#endif

#include <hip/hip_runtime.h>

#ifndef BENCHMARK_CONFIG_TUNING
    #include <rocprim/device/config_types.hpp>
    #include <rocprim/types.hpp>
#endif

#include <cstddef>
#include <string>
#include <vector>
#ifndef BENCHMARK_CONFIG_TUNING
    #include <stdint.h>
#endif

#define CREATE_PARTITION_FLAG_BENCHMARK(T, F, p) \
    executor.queue<device_partition_flag_benchmark<T, rocprim::default_config, F, p>>();

#define CREATE_PARTITION_PREDICATE_BENCHMARK(T, p) \
    executor.queue<device_partition_predicate_benchmark<T, rocprim::default_config, p>>();

#define CREATE_PARTITION_TWO_WAY_FLAG_BENCHMARK(T, F, p) \
    executor.queue<device_partition_two_way_flag_benchmark<T, rocprim::default_config, F, p>>();

#define CREATE_PARTITION_TWO_WAY_PREDICATE_BENCHMARK(T, p) \
    executor.queue<device_partition_two_way_predicate_benchmark<T, rocprim::default_config, p>>();

#define CREATE_PARTITION_THREE_WAY_BENCHMARK(T, p) \
    executor.queue<device_partition_three_way_benchmark<T, rocprim::default_config, p>>();

#define BENCHMARK_FLAG_TYPE(type, flag_type)                                      \
    CREATE_PARTITION_FLAG_BENCHMARK(type, flag_type, partition_probability::p005) \
    CREATE_PARTITION_FLAG_BENCHMARK(type, flag_type, partition_probability::p025) \
    CREATE_PARTITION_FLAG_BENCHMARK(type, flag_type, partition_probability::p050) \
    CREATE_PARTITION_FLAG_BENCHMARK(type, flag_type, partition_probability::p075)

#define BENCHMARK_PREDICATE_TYPE(type)                                      \
    CREATE_PARTITION_PREDICATE_BENCHMARK(type, partition_probability::p005) \
    CREATE_PARTITION_PREDICATE_BENCHMARK(type, partition_probability::p025) \
    CREATE_PARTITION_PREDICATE_BENCHMARK(type, partition_probability::p050) \
    CREATE_PARTITION_PREDICATE_BENCHMARK(type, partition_probability::p075)

#define BENCHMARK_TWO_WAY_FLAG_TYPE(type, flag_type)                                      \
    CREATE_PARTITION_TWO_WAY_FLAG_BENCHMARK(type, flag_type, partition_probability::p005) \
    CREATE_PARTITION_TWO_WAY_FLAG_BENCHMARK(type, flag_type, partition_probability::p025) \
    CREATE_PARTITION_TWO_WAY_FLAG_BENCHMARK(type, flag_type, partition_probability::p050) \
    CREATE_PARTITION_TWO_WAY_FLAG_BENCHMARK(type, flag_type, partition_probability::p075)

#define BENCHMARK_TWO_WAY_PREDICATE_TYPE(type)                                      \
    CREATE_PARTITION_TWO_WAY_PREDICATE_BENCHMARK(type, partition_probability::p005) \
    CREATE_PARTITION_TWO_WAY_PREDICATE_BENCHMARK(type, partition_probability::p025) \
    CREATE_PARTITION_TWO_WAY_PREDICATE_BENCHMARK(type, partition_probability::p050) \
    CREATE_PARTITION_TWO_WAY_PREDICATE_BENCHMARK(type, partition_probability::p075)

#define BENCHMARK_THREE_WAY_TYPE(type)                                                     \
    CREATE_PARTITION_THREE_WAY_BENCHMARK(type, partition_three_way_probability::p005_p025) \
    CREATE_PARTITION_THREE_WAY_BENCHMARK(type, partition_three_way_probability::p025_p050) \
    CREATE_PARTITION_THREE_WAY_BENCHMARK(type, partition_three_way_probability::p050_p075) \
    CREATE_PARTITION_THREE_WAY_BENCHMARK(type, partition_three_way_probability::p075_p100)

#define BENCHMARK_TYPES_TUNING(T)          \
    BENCHMARK_FLAG_TYPE(T, int8_t)         \
    BENCHMARK_PREDICATE_TYPE(T)            \
    BENCHMARK_TWO_WAY_FLAG_TYPE(T, int8_t) \
    BENCHMARK_TWO_WAY_PREDICATE_TYPE(T)    \
    BENCHMARK_THREE_WAY_TYPE(T)

int main(int argc, char* argv[])
{
    primbench::settings settings;
    settings.size = 128 * primbench::MiB;
    primbench::executor executor(argc, argv, settings);

#ifndef BENCHMARK_CONFIG_TUNING
    // Tuned types
    BENCHMARK_TYPES_TUNING(rocprim::int128_t)
    BENCHMARK_TYPES_TUNING(int64_t)
    BENCHMARK_TYPES_TUNING(int32_t)
    BENCHMARK_TYPES_TUNING(int16_t)
    BENCHMARK_TYPES_TUNING(int8_t)
    BENCHMARK_TYPES_TUNING(double)
    BENCHMARK_TYPES_TUNING(float)
    BENCHMARK_TYPES_TUNING(rocprim::half)

    #ifndef BENCHMARK_AUTOTUNED_TYPES_ONLY
    // Not tuned types
    BENCHMARK_FLAG_TYPE(uint8_t, uint8_t)
    BENCHMARK_FLAG_TYPE(rocprim::uint128_t, uint8_t)

    BENCHMARK_PREDICATE_TYPE(uint8_t)
    BENCHMARK_PREDICATE_TYPE(rocprim::uint128_t)

    BENCHMARK_TWO_WAY_FLAG_TYPE(uint8_t, uint8_t)
    BENCHMARK_TWO_WAY_FLAG_TYPE(rocprim::uint128_t, uint8_t)

    BENCHMARK_TWO_WAY_PREDICATE_TYPE(uint8_t)
    BENCHMARK_TWO_WAY_PREDICATE_TYPE(rocprim::uint128_t)

    BENCHMARK_THREE_WAY_TYPE(uint8_t)
    BENCHMARK_THREE_WAY_TYPE(rocprim::uint128_t)

    // Not tuned custom types
    BENCHMARK_FLAG_TYPE(custom_f64_f64, uint8_t)
    BENCHMARK_FLAG_TYPE(huge_1024_f32_f32, uint8_t)

    BENCHMARK_PREDICATE_TYPE(custom_i32_f64)
    BENCHMARK_PREDICATE_TYPE(huge_1024_f32_f32)

    BENCHMARK_TWO_WAY_FLAG_TYPE(custom_f64_f64, uint8_t)
    BENCHMARK_TWO_WAY_FLAG_TYPE(huge_1024_f32_f32, uint8_t)

    BENCHMARK_TWO_WAY_PREDICATE_TYPE(custom_i32_f64)
    BENCHMARK_TWO_WAY_PREDICATE_TYPE(huge_1024_f32_f32)

    BENCHMARK_THREE_WAY_TYPE(custom_i32_f64)
    BENCHMARK_THREE_WAY_TYPE(huge_1024_f32_f32)
    #endif
#endif

    executor.run();
}
