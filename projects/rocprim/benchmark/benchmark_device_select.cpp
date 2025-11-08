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

#include "benchmark_device_select.parallel.hpp"
#include "benchmark_utils.hpp"

#define CREATE_SELECT_PREDICATED_FLAG_BENCHMARK(T, F, p) \
    executor.queue_instance(                             \
        device_select_predicated_flag_benchmark<T, F, rocprim::default_config, p>());

#define CREATE_SELECT_FLAG_BENCHMARK(T, F, p) \
    executor.queue_instance(device_select_flag_benchmark<T, rocprim::default_config, F, p>());

#define CREATE_SELECT_PREDICATE_BENCHMARK(T, p) \
    executor.queue_instance(device_select_predicate_benchmark<T, rocprim::default_config, p>());

#define CREATE_UNIQUE_BENCHMARK(T, p) \
    executor.queue_instance(device_select_unique_benchmark<T, rocprim::default_config, p>());

#define CREATE_UNIQUE_BY_KEY_BENCHMARK(K, V, p) \
    executor.queue_instance(                    \
        device_select_unique_by_key_benchmark<K, V, rocprim::default_config, p>());

#define BENCHMARK_SELECT_PREDICATED_FLAG_TYPE(type, value)                         \
    CREATE_SELECT_PREDICATED_FLAG_BENCHMARK(type, value, select_probability::p005) \
    CREATE_SELECT_PREDICATED_FLAG_BENCHMARK(type, value, select_probability::p025) \
    CREATE_SELECT_PREDICATED_FLAG_BENCHMARK(type, value, select_probability::p050) \
    CREATE_SELECT_PREDICATED_FLAG_BENCHMARK(type, value, select_probability::p075)

#define BENCHMARK_SELECT_FLAG_TYPE(type, value)                         \
    CREATE_SELECT_FLAG_BENCHMARK(type, value, select_probability::p005) \
    CREATE_SELECT_FLAG_BENCHMARK(type, value, select_probability::p025) \
    CREATE_SELECT_FLAG_BENCHMARK(type, value, select_probability::p050) \
    CREATE_SELECT_FLAG_BENCHMARK(type, value, select_probability::p075)

#define BENCHMARK_SELECT_PREDICATE_TYPE(type)                         \
    CREATE_SELECT_PREDICATE_BENCHMARK(type, select_probability::p005) \
    CREATE_SELECT_PREDICATE_BENCHMARK(type, select_probability::p025) \
    CREATE_SELECT_PREDICATE_BENCHMARK(type, select_probability::p050) \
    CREATE_SELECT_PREDICATE_BENCHMARK(type, select_probability::p075)

#define BENCHMARK_UNIQUE_TYPE(type)                         \
    CREATE_UNIQUE_BENCHMARK(type, select_probability::p005) \
    CREATE_UNIQUE_BENCHMARK(type, select_probability::p025) \
    CREATE_UNIQUE_BENCHMARK(type, select_probability::p050) \
    CREATE_UNIQUE_BENCHMARK(type, select_probability::p075)

#define BENCHMARK_UNIQUE_BY_KEY_TYPE(K, V)                         \
    CREATE_UNIQUE_BY_KEY_BENCHMARK(K, V, select_probability::p005) \
    CREATE_UNIQUE_BY_KEY_BENCHMARK(K, V, select_probability::p025) \
    CREATE_UNIQUE_BY_KEY_BENCHMARK(K, V, select_probability::p050) \
    CREATE_UNIQUE_BY_KEY_BENCHMARK(K, V, select_probability::p075)

#define BENCHMARK_TYPE_TUNING(KEY_TYPE, VALUE_TYPE)    \
    BENCHMARK_UNIQUE_BY_KEY_TYPE(KEY_TYPE, VALUE_TYPE) \
    BENCHMARK_SELECT_PREDICATED_FLAG_TYPE(KEY_TYPE, VALUE_TYPE)

#define BENCHMARK_TYPES_TUNING(KEY_TYPE)               \
    BENCHMARK_SELECT_FLAG_TYPE(KEY_TYPE, char)         \
    BENCHMARK_SELECT_PREDICATE_TYPE(KEY_TYPE)          \
    BENCHMARK_UNIQUE_TYPE(KEY_TYPE)                    \
    BENCHMARK_TYPE_TUNING(KEY_TYPE, rocprim::int128_t) \
    BENCHMARK_TYPE_TUNING(KEY_TYPE, int64_t)           \
    BENCHMARK_TYPE_TUNING(KEY_TYPE, int)               \
    BENCHMARK_TYPE_TUNING(KEY_TYPE, short)             \
    BENCHMARK_TYPE_TUNING(KEY_TYPE, int8_t)

int main(int argc, char* argv[])
{
    benchmark_utils::executor executor(argc, argv, 128 * benchmark_utils::MiB, 10, 5);

#ifndef BENCHMARK_CONFIG_TUNING
    // Tuned types
    BENCHMARK_TYPES_TUNING(rocprim::int128_t)
    BENCHMARK_TYPES_TUNING(int64_t)
    BENCHMARK_TYPES_TUNING(int)
    BENCHMARK_TYPES_TUNING(short)
    BENCHMARK_TYPES_TUNING(int8_t)
    BENCHMARK_TYPES_TUNING(double)
    BENCHMARK_TYPES_TUNING(float)
    BENCHMARK_TYPES_TUNING(rocprim::half)

    #ifndef BENCHMARK_AUTOTUNED_TYPES_ONLY
    // Not tuned types
    BENCHMARK_SELECT_FLAG_TYPE(uint8_t, uint8_t)
    BENCHMARK_SELECT_FLAG_TYPE(rocprim::uint128_t, unsigned char)

    BENCHMARK_SELECT_PREDICATE_TYPE(uint8_t)
    BENCHMARK_SELECT_PREDICATE_TYPE(rocprim::uint128_t)

    BENCHMARK_SELECT_PREDICATED_FLAG_TYPE(uint8_t, uint8_t)
    BENCHMARK_SELECT_PREDICATED_FLAG_TYPE(rocprim::uint128_t, unsigned char)

    BENCHMARK_UNIQUE_TYPE(uint8_t)
    BENCHMARK_UNIQUE_TYPE(rocprim::uint128_t)

    BENCHMARK_UNIQUE_BY_KEY_TYPE(float, double)
    BENCHMARK_UNIQUE_BY_KEY_TYPE(uint8_t, uint8_t)
    BENCHMARK_UNIQUE_BY_KEY_TYPE(int8_t, double)
    BENCHMARK_UNIQUE_BY_KEY_TYPE(rocprim::half, rocprim::half)
    BENCHMARK_UNIQUE_BY_KEY_TYPE(rocprim::uint128_t, rocprim::int128_t)

    // Not tuned custom types
    using custom_double2    = common::custom_type<double, double>;
    using custom_int_double = common::custom_type<int, double>;
    using huge_float2       = common::custom_huge_type<1024, float, float>;

    BENCHMARK_SELECT_FLAG_TYPE(custom_double2, unsigned char)
    BENCHMARK_SELECT_PREDICATED_FLAG_TYPE(custom_double2, unsigned char)
    BENCHMARK_UNIQUE_BY_KEY_TYPE(double, custom_double2)

    BENCHMARK_SELECT_PREDICATE_TYPE(custom_int_double)
    BENCHMARK_UNIQUE_TYPE(custom_int_double)
    BENCHMARK_UNIQUE_BY_KEY_TYPE(custom_int_double, custom_int_double)

    BENCHMARK_SELECT_FLAG_TYPE(huge_float2, unsigned char)
    BENCHMARK_SELECT_PREDICATE_TYPE(huge_float2)
    BENCHMARK_SELECT_PREDICATED_FLAG_TYPE(huge_float2, unsigned char)
    BENCHMARK_UNIQUE_TYPE(huge_float2)
    BENCHMARK_UNIQUE_BY_KEY_TYPE(huge_float2, huge_float2)
    #endif
#endif

    executor.run();
}
