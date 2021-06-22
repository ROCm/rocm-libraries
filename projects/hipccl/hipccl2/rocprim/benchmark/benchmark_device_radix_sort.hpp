// MIT License
//
// Copyright (c) 2017-2021 Advanced Micro Devices, Inc. All rights reserved.
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

#include <iostream>
#include <chrono>
#include <vector>
#include <limits>
#include <memory>
#include <string>
#include <cstdio>
#include <cstdlib>

// Google Benchmark
#include "benchmark/benchmark.h"
// CmdParser
#include "cmdparser.hpp"
#include "benchmark_utils.hpp"

// HIP API
#include <hip/hip_runtime.h>

// rocPRIM
#include <rocprim/rocprim.hpp>

#define HIP_CHECK(condition)         \
  {                                  \
    hipError_t error = condition;    \
    if(error != hipSuccess){         \
        std::cout << "HIP error: " << error << " line: " << __LINE__ << std::endl; \
        exit(error); \
    } \
  }

template <typename T>
struct Traits
{
    static const char * TYPE_NAME;
};
// Generic definition as a fall-back:
template <typename T> const char * Traits<T>::TYPE_NAME = "unknown";

// Explicit definitions
template < > const char * Traits< int >::TYPE_NAME = "int";
template < > const char * Traits< short >::TYPE_NAME = "short";
template < > const char * Traits< int8_t >::TYPE_NAME = "int8_t";
template < > const char * Traits< uint8_t >::TYPE_NAME = "uint8_t";
template < > const char * Traits< rocprim::half >::TYPE_NAME = "rocprim::half";
template < > const char * Traits< long long >::TYPE_NAME = "long long";
template < > const char * Traits< float >::TYPE_NAME = "float";
template < > const char * Traits< double >::TYPE_NAME = "double";
template < > const char * Traits< float2 >::TYPE_NAME = "float2";
template < > const char * Traits< double2 >::TYPE_NAME = "double2";
template < > const char * Traits< custom_type<float, float> >::TYPE_NAME = "custom_float2";
template < > const char * Traits< custom_type<double, double> >::TYPE_NAME = "custom_double2";

namespace rp = rocprim;

const unsigned int batch_size = 10;
const unsigned int warmup_size = 5;

template<class Key>
std::vector<Key> generate_keys(size_t size)
{
    using key_type = Key;

    if(std::is_floating_point<key_type>::value)
    {
        return get_random_data<key_type>(size, (key_type)-1000, (key_type)+1000, size);
    }
    else
    {
        return get_random_data<key_type>(
            size,
            std::numeric_limits<key_type>::min(),
            std::numeric_limits<key_type>::max(),
            size
        );
    }
}

template<class Key, class Config>
void run_sort_keys_benchmark(benchmark::State& state,
                             hipStream_t stream,
                             size_t size)
{
    auto keys_input = generate_keys<Key>(size);

    using key_type = Key;

    key_type * d_keys_input;
    key_type * d_keys_output;
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_keys_input), size * sizeof(key_type)));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_keys_output), size * sizeof(key_type)));

    HIP_CHECK(
        hipMemcpy(
            d_keys_input, keys_input.data(),
            size * sizeof(key_type),
            hipMemcpyHostToDevice
        )
    );

    void * d_temporary_storage = nullptr;
    size_t temporary_storage_bytes = 0;
    HIP_CHECK(
        rp::radix_sort_keys<Config>(
            d_temporary_storage, temporary_storage_bytes,
            d_keys_input, d_keys_output, size,
            0, sizeof(key_type) * 8,
            stream, false
        )
    );

    HIP_CHECK(hipMalloc(&d_temporary_storage, temporary_storage_bytes));
    HIP_CHECK(hipDeviceSynchronize());

    // Warm-up
    for(size_t i = 0; i < warmup_size; i++)
    {
        HIP_CHECK(
            rp::radix_sort_keys<Config>(
                d_temporary_storage, temporary_storage_bytes,
                d_keys_input, d_keys_output, size,
                0, sizeof(key_type) * 8,
                stream, false
            )
        );
    }
    HIP_CHECK(hipDeviceSynchronize());

    for (auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();

        for(size_t i = 0; i < batch_size; i++)
        {
            HIP_CHECK(
                rp::radix_sort_keys<Config>(
                    d_temporary_storage, temporary_storage_bytes,
                    d_keys_input, d_keys_output, size,
                    0, sizeof(key_type) * 8,
                    stream, false
                )
            );
        }
        HIP_CHECK(hipDeviceSynchronize());

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }
    state.SetBytesProcessed(state.iterations() * batch_size * size * sizeof(key_type));
    state.SetItemsProcessed(state.iterations() * batch_size * size);

    HIP_CHECK(hipFree(d_temporary_storage));
    HIP_CHECK(hipFree(d_keys_input));
    HIP_CHECK(hipFree(d_keys_output));
}

template<class Key, class Value, class Config>
void run_sort_pairs_benchmark(benchmark::State& state,
                              hipStream_t stream,
                              size_t size)
{
    auto keys_input = generate_keys<Key>(size);

    using key_type = Key;
    using value_type = Value;

    std::vector<value_type> values_input(size);
    for(size_t i = 0; i < size; i++)
    {
        values_input[i] = value_type(i);
    }

    key_type * d_keys_input;
    key_type * d_keys_output;
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_keys_input), size * sizeof(key_type)));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_keys_output), size * sizeof(key_type)));
    HIP_CHECK(
        hipMemcpy(
            d_keys_input, keys_input.data(),
            size * sizeof(key_type),
            hipMemcpyHostToDevice
        )
    );

    value_type * d_values_input;
    value_type * d_values_output;
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_values_input), size * sizeof(value_type)));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_values_output), size * sizeof(value_type)));
    HIP_CHECK(
        hipMemcpy(
            d_values_input, values_input.data(),
            size * sizeof(value_type),
            hipMemcpyHostToDevice
        )
    );

    void * d_temporary_storage = nullptr;
    size_t temporary_storage_bytes = 0;
    HIP_CHECK(
        rp::radix_sort_pairs<Config>(
            d_temporary_storage, temporary_storage_bytes,
            d_keys_input, d_keys_output, d_values_input, d_values_output, size,
            0, sizeof(key_type) * 8,
            stream, false
        )
    );

    HIP_CHECK(hipMalloc(&d_temporary_storage, temporary_storage_bytes));
    HIP_CHECK(hipDeviceSynchronize());

    // Warm-up
    for(size_t i = 0; i < warmup_size; i++)
    {
        HIP_CHECK(
            rp::radix_sort_pairs<Config>(
                d_temporary_storage, temporary_storage_bytes,
                d_keys_input, d_keys_output, d_values_input, d_values_output, size,
                0, sizeof(key_type) * 8,
                stream, false
            )
        );
    }
    HIP_CHECK(hipDeviceSynchronize());

    for (auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();

        for(size_t i = 0; i < batch_size; i++)
        {
            HIP_CHECK(
                rp::radix_sort_pairs<Config>(
                    d_temporary_storage, temporary_storage_bytes,
                    d_keys_input, d_keys_output, d_values_input, d_values_output, size,
                    0, sizeof(key_type) * 8,
                    stream, false
                )
            );
        }
        HIP_CHECK(hipDeviceSynchronize());

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }
    state.SetBytesProcessed(
        state.iterations() * batch_size * size * (sizeof(key_type) + sizeof(value_type))
    );
    state.SetItemsProcessed(state.iterations() * batch_size * size);

    HIP_CHECK(hipFree(d_temporary_storage));
    HIP_CHECK(hipFree(d_keys_input));
    HIP_CHECK(hipFree(d_keys_output));
    HIP_CHECK(hipFree(d_values_input));
    HIP_CHECK(hipFree(d_values_output));
}

#ifndef BENCHMARK_CONFIG_TUNING

#define CREATE_SORT_KEYS_BENCHMARK(Key) \
    { \
        benchmarks.push_back( \
            benchmark::RegisterBenchmark( \
                (std::string("sort_keys") + "<" #Key ">").c_str(), \
                [=](benchmark::State& state) { run_sort_keys_benchmark<Key, rocprim::default_config>(state, stream, size); } \
            ) \
        ); \
    }

#define CREATE_SORT_PAIRS_BENCHMARK(Key, Value) \
    { \
        benchmarks.push_back( \
            benchmark::RegisterBenchmark( \
                (std::string("sort_pairs") + "<" #Key ", " #Value">").c_str(), \
                [=](benchmark::State& state) { run_sort_pairs_benchmark<Key, Value, rocprim::default_config>(state, stream, size); } \
            ) \
        ); \
    }

    // Compilation may never finish, if the compiler needs to compile too many kernels,
    // it is recommended to compile benchmarks only for 1-2 types when BENCHMARK_CONFIG_TUNING is used
    // (all other CREATE_*_BENCHMARK should be commented/removed).
    void add_sort_keys_benchmarks(std::vector<benchmark::internal::Benchmark*>& benchmarks,
                                  hipStream_t stream,
                                  size_t size)
    {
        CREATE_SORT_KEYS_BENCHMARK(int)
        CREATE_SORT_KEYS_BENCHMARK(float)
        CREATE_SORT_KEYS_BENCHMARK(long long)
        CREATE_SORT_KEYS_BENCHMARK(int8_t)
        CREATE_SORT_KEYS_BENCHMARK(uint8_t)
        CREATE_SORT_KEYS_BENCHMARK(rocprim::half)
        CREATE_SORT_KEYS_BENCHMARK(short)
    }

    void add_sort_pairs_benchmarks(std::vector<benchmark::internal::Benchmark*>& benchmarks,
                                   hipStream_t stream,
                                   size_t size)
    {
        using custom_float2 = custom_type<float, float>;
        using custom_double2 = custom_type<double, double>;

        CREATE_SORT_PAIRS_BENCHMARK(int, float)
        CREATE_SORT_PAIRS_BENCHMARK(int, double)
        CREATE_SORT_PAIRS_BENCHMARK(int, float2)
        CREATE_SORT_PAIRS_BENCHMARK(int, custom_float2)
        CREATE_SORT_PAIRS_BENCHMARK(int, double2)
        CREATE_SORT_PAIRS_BENCHMARK(int, custom_double2)

        CREATE_SORT_PAIRS_BENCHMARK(long long, float)
        CREATE_SORT_PAIRS_BENCHMARK(long long, double)
        CREATE_SORT_PAIRS_BENCHMARK(long long, float2)
        CREATE_SORT_PAIRS_BENCHMARK(long long, custom_float2)
        CREATE_SORT_PAIRS_BENCHMARK(long long, double2)
        CREATE_SORT_PAIRS_BENCHMARK(long long, custom_double2)
        CREATE_SORT_PAIRS_BENCHMARK(int8_t, int8_t)
        CREATE_SORT_PAIRS_BENCHMARK(uint8_t, uint8_t)
        CREATE_SORT_PAIRS_BENCHMARK(rocprim::half, rocprim::half)
    }

#endif // BENCHMARK_CONFIG_TUNING
