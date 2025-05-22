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

#include "benchmark_device_partition.parallel.hpp"
#include "benchmark_utils.hpp"

#ifndef BENCHMARK_CONFIG_TUNING
    #include "../common/utils_custom_type.hpp"
#endif

// HIP API
#include <hip/hip_runtime.h>

// rocPRIM
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
    executor.queue_instance(device_partition_flag_benchmark<T, rocprim::default_config, F, p>());

#define CREATE_PARTITION_PREDICATE_BENCHMARK(T, p) \
    executor.queue_instance(device_partition_predicate_benchmark<T, rocprim::default_config, p>());

#define CREATE_PARTITION_TWO_WAY_FLAG_BENCHMARK(T, F, p) \
    executor.queue_instance(                             \
        device_partition_two_way_flag_benchmark<T, rocprim::default_config, F, p>());

#define CREATE_PARTITION_TWO_WAY_PREDICATE_BENCHMARK(T, p) \
    executor.queue_instance(                               \
        device_partition_two_way_predicate_benchmark<T, rocprim::default_config, p>());

#define CREATE_PARTITION_THREE_WAY_BENCHMARK(T, p) \
    executor.queue_instance(device_partition_three_way_benchmark<T, rocprim::default_config, p>());

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

int main(int argc, char* argv[])
{
    benchmark_utils::executor executor(argc, argv, 128 * benchmark_utils::MiB, 10, 5);

#ifndef BENCHMARK_CONFIG_TUNING
    using custom_double2    = common::custom_type<double, double>;
    using custom_int_double = common::custom_type<int, double>;
    using huge_float2       = common::custom_huge_type<1024, float, float>;

    BENCHMARK_FLAG_TYPE(int, unsigned char)
    BENCHMARK_FLAG_TYPE(float, unsigned char)
    BENCHMARK_FLAG_TYPE(double, unsigned char)
    BENCHMARK_FLAG_TYPE(uint8_t, uint8_t)
    BENCHMARK_FLAG_TYPE(int8_t, int8_t)
    BENCHMARK_FLAG_TYPE(rocprim::half, int8_t)
    BENCHMARK_FLAG_TYPE(custom_double2, unsigned char)
    BENCHMARK_FLAG_TYPE(rocprim::int128_t, int8_t)
    BENCHMARK_FLAG_TYPE(rocprim::uint128_t, uint8_t)
    BENCHMARK_FLAG_TYPE(huge_float2, uint8_t)

    BENCHMARK_PREDICATE_TYPE(int)
    BENCHMARK_PREDICATE_TYPE(float)
    BENCHMARK_PREDICATE_TYPE(double)
    BENCHMARK_PREDICATE_TYPE(uint8_t)
    BENCHMARK_PREDICATE_TYPE(int8_t)
    BENCHMARK_PREDICATE_TYPE(rocprim::half)
    BENCHMARK_PREDICATE_TYPE(custom_int_double)
    BENCHMARK_PREDICATE_TYPE(rocprim::int128_t)
    BENCHMARK_PREDICATE_TYPE(rocprim::uint128_t)
    BENCHMARK_PREDICATE_TYPE(huge_float2)

    BENCHMARK_TWO_WAY_FLAG_TYPE(int, unsigned char)
    BENCHMARK_TWO_WAY_FLAG_TYPE(float, unsigned char)
    BENCHMARK_TWO_WAY_FLAG_TYPE(double, unsigned char)
    BENCHMARK_TWO_WAY_FLAG_TYPE(uint8_t, uint8_t)
    BENCHMARK_TWO_WAY_FLAG_TYPE(int8_t, int8_t)
    BENCHMARK_TWO_WAY_FLAG_TYPE(rocprim::half, int8_t)
    BENCHMARK_TWO_WAY_FLAG_TYPE(custom_double2, unsigned char)
    BENCHMARK_TWO_WAY_FLAG_TYPE(rocprim::int128_t, int8_t)
    BENCHMARK_TWO_WAY_FLAG_TYPE(rocprim::uint128_t, uint8_t)
    BENCHMARK_TWO_WAY_FLAG_TYPE(huge_float2, uint8_t)

    BENCHMARK_TWO_WAY_PREDICATE_TYPE(int)
    BENCHMARK_TWO_WAY_PREDICATE_TYPE(float)
    BENCHMARK_TWO_WAY_PREDICATE_TYPE(double)
    BENCHMARK_TWO_WAY_PREDICATE_TYPE(uint8_t)
    BENCHMARK_TWO_WAY_PREDICATE_TYPE(int8_t)
    BENCHMARK_TWO_WAY_PREDICATE_TYPE(rocprim::half)
    BENCHMARK_TWO_WAY_PREDICATE_TYPE(custom_int_double)
    BENCHMARK_TWO_WAY_PREDICATE_TYPE(rocprim::int128_t)
    BENCHMARK_TWO_WAY_PREDICATE_TYPE(rocprim::uint128_t)
    BENCHMARK_TWO_WAY_PREDICATE_TYPE(huge_float2)

    BENCHMARK_THREE_WAY_TYPE(int)
    BENCHMARK_THREE_WAY_TYPE(float)
    BENCHMARK_THREE_WAY_TYPE(double)
    BENCHMARK_THREE_WAY_TYPE(uint8_t)
    BENCHMARK_THREE_WAY_TYPE(int8_t)
    BENCHMARK_THREE_WAY_TYPE(rocprim::half)
    BENCHMARK_THREE_WAY_TYPE(custom_int_double)
    BENCHMARK_THREE_WAY_TYPE(rocprim::int128_t)
    BENCHMARK_THREE_WAY_TYPE(rocprim::uint128_t)
    BENCHMARK_THREE_WAY_TYPE(huge_float2)
#endif

    executor.run();
}
