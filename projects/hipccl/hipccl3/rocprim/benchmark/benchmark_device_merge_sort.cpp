// MIT License
//
// Copyright (c) 2017-2019 Advanced Micro Devices, Inc. All rights reserved.
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

#include <algorithm>
#include <iostream>
#include <chrono>
#include <vector>
#include <limits>
#include <utility>
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

#ifndef DEFAULT_N
const size_t DEFAULT_N = 1024 * 1024 * 32;
#endif

namespace rp = rocprim;

const unsigned int batch_size = 10;
const unsigned int warmup_size = 5;

template<class Key, class Config = rp::default_config>
void run_sort_keys_benchmark(benchmark::State& state, hipStream_t stream, size_t size)
{
    using key_type = Key;

    // Generate data
    std::vector<key_type> keys_input;
    if(std::is_floating_point<key_type>::value)
    {
        keys_input = get_random_data<key_type>(size, (key_type)-1000, (key_type)+1000);
    }
    else
    {
        keys_input = get_random_data<key_type>(
            size,
            std::numeric_limits<key_type>::min(),
            std::numeric_limits<key_type>::max()
        );
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

    ::rocprim::less<key_type> lesser_op;

    void * d_temporary_storage = nullptr;
    size_t temporary_storage_bytes = 0;
    HIP_CHECK(
        rp::merge_sort<Config>(
            d_temporary_storage, temporary_storage_bytes,
            d_keys_input, d_keys_output, size,
            lesser_op, stream, false
        )
    );

    HIP_CHECK(hipMalloc(&d_temporary_storage, temporary_storage_bytes));
    HIP_CHECK(hipDeviceSynchronize());

    // Warm-up
    for(size_t i = 0; i < warmup_size; i++)
    {
        HIP_CHECK(
            rp::merge_sort<Config>(
                d_temporary_storage, temporary_storage_bytes,
                d_keys_input, d_keys_output, size,
                lesser_op, stream, false
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
                rp::merge_sort<Config>(
                    d_temporary_storage, temporary_storage_bytes,
                    d_keys_input, d_keys_output, size,
                    lesser_op, stream, false
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

template<class Key, class Value, class Config = rp::default_config>
void run_sort_pairs_benchmark(benchmark::State& state, hipStream_t stream, size_t size)
{
    using key_type = Key;
    using value_type = Value;

    // Generate data
    std::vector<key_type> keys_input;
    if(std::is_floating_point<key_type>::value)
    {
        keys_input = get_random_data<key_type>(size, (key_type)-1000, (key_type)+1000);
    }
    else
    {
        keys_input = get_random_data<key_type>(
            size,
            std::numeric_limits<key_type>::min(),
            std::numeric_limits<key_type>::max()
        );
    }

    std::vector<value_type> values_input(size);
    std::iota(values_input.begin(), values_input.end(), 0);

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

    ::rocprim::less<key_type> lesser_op;

    void * d_temporary_storage = nullptr;
    size_t temporary_storage_bytes = 0;
    HIP_CHECK(
        rp::merge_sort<Config>(
            d_temporary_storage, temporary_storage_bytes,
            d_keys_input, d_keys_output, d_values_input, d_values_output, size,
            lesser_op, stream, false
        )
    );

    HIP_CHECK(hipMalloc(&d_temporary_storage, temporary_storage_bytes));
    HIP_CHECK(hipDeviceSynchronize());

    // Warm-up
    for(size_t i = 0; i < warmup_size; i++)
    {
        HIP_CHECK(
            rp::merge_sort<Config>(
                d_temporary_storage, temporary_storage_bytes,
                d_keys_input, d_keys_output, d_values_input, d_values_output, size,
                lesser_op, stream, false
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
                rp::merge_sort<Config>(
                    d_temporary_storage, temporary_storage_bytes,
                    d_keys_input, d_keys_output, d_values_input, d_values_output, size,
                    lesser_op, stream, false
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

#ifdef BENCHMARK_CONFIG_TUNING

template <typename T>
struct Traits
{
    static const char* name;
};
// Generic definition as a fall-back:
template <typename T>
const char* Traits<T>::name = "unknown";

// Explicit definitions
template <>
const char* Traits<int>::name = "int";
template <>
const char* Traits<short>::name = "short";
template <>
const char* Traits<int8_t>::name = "int8_t";
template <>
const char* Traits<uint8_t>::name = "uint8_t";
template <>
const char* Traits<rocprim::half>::name = "rocprim::half";
template <>
const char* Traits<long long>::name = "long long";
template <>
const char* Traits<float>::name = "float";
template <>
const char* Traits<double>::name = "double";
template <>
const char* Traits<custom_type<int, int>>::name = "custom_int2";
template <>
const char* Traits<custom_type<float, float>>::name = "custom_float2";
template <>
const char* Traits<custom_type<double, double>>::name = "custom_double2";
template <>
const char* Traits<custom_type<char, double>>::name = "custom_char_double";
template <>
const char* Traits<custom_type<long long, double>>::name = "custom_longlong_double";

template <typename T, T, typename>
struct make_index_range_impl;

template <typename T, T Start, T... I>
struct make_index_range_impl<T, Start, std::integer_sequence<T, I...>>
{
    using type = std::integer_sequence<T, (Start + I)...>;
};

// make a std::integer_sequence with values from Start to End inclusive
template <typename T, T Start, T End>
using make_index_range =
    typename make_index_range_impl<T, Start, std::make_integer_sequence<T, End - Start + 1>>::type;

template <typename T, template <T> class Function, T... I, typename... Args>
void static_for_each_impl(std::integer_sequence<T, I...>, Args... args)
{
    int a[] = {(Function<I> {}(args...), 0)...};
    static_cast<void>(a);
}

// call the supplied template with all values of the std::integer_sequence Indices
template <typename Indices,
          template <typename Indices::value_type>
          class Function,
          typename... Args>
void static_for_each(Args... args)
{
    static_for_each_impl<typename Indices::value_type, Function>(Indices {}, args...);
}

template <class Key, class Value>
struct name_prefix_fn
{
    auto operator()()
    {
        return std::string {"sort_pairs<"} + Traits<Key>::name + ", " + Traits<Value>::name;
    };
};

template <class Key>
struct name_prefix_fn<Key, rp::empty_type>
{
    auto operator()()
    {
        return std::string {"sort_keys<"} + Traits<Key>::name;
    };
};

template <class Key, class Value>
struct select_benchmark_function
{
    template <typename Config>
    static void run(benchmark::State& state, hipStream_t stream, size_t size)
    {
        run_sort_pairs_benchmark<Key, Value, Config>(state, stream, size);
    }
};

template <class Key>
struct select_benchmark_function<Key, rp::empty_type>
{
    template <typename Config>
    static void run(benchmark::State& state, hipStream_t stream, size_t size)
    {
        run_sort_keys_benchmark<Key, Config>(state, stream, size);
    }
};

template <class Key, class Value = rp::empty_type>
struct create_benchmarks
{
    template <unsigned int MergeBlockSizeExponent>
    struct sweep_config_3d
    {
        template <unsigned int SortBlockSizeExponent>
        struct sweep_config_2d
        {
            template <unsigned int SortItemsPerThreadExponent>
            struct sweep_config
            {
                constexpr static auto merge_block_size      = 1u << MergeBlockSizeExponent;
                constexpr static auto sort_block_size       = 1u << SortBlockSizeExponent;
                constexpr static auto sort_items_per_thread = 1u << SortItemsPerThreadExponent;

                std::string get_name()
                {
                    return name_prefix_fn<Key, Value> {}() + ", " + std::to_string(merge_block_size)
                           + ", " + std::to_string(sort_block_size) + ", "
                           + std::to_string(sort_items_per_thread) + ">";
                }

                void operator()(std::vector<benchmark::internal::Benchmark*>& benchmarks,
                                hipStream_t                                   stream,
                                size_t                                        size)
                {
                    using config = rp::
                        merge_sort_config<merge_block_size, sort_block_size, sort_items_per_thread>;

                    benchmarks.emplace_back(benchmark::RegisterBenchmark(
                        get_name().c_str(), [=](benchmark::State& state) {
                            select_benchmark_function<Key, Value>::template run<config>(
                                state, stream, size);
                        }));
                }
            };

            void operator()(std::vector<benchmark::internal::Benchmark*>& benchmarks,
                            hipStream_t                                   stream,
                            size_t                                        size)
            {
                // Sort items per block must be divisible by merge_block_size, so make
                // the items per thread at least as large that the items_per_block
                // is equal to merge_block_size.
                static constexpr auto min_items_per_thread
                    = MergeBlockSizeExponent
                      - std::min(SortBlockSizeExponent, MergeBlockSizeExponent);

                // Very large block sizes don't work with large items_per_blocks since
                // shared memory is limited
                static constexpr auto max_items_per_thread
                    = std::min(4u, 11u - SortBlockSizeExponent);

                static_for_each<
                    make_index_range<unsigned int, min_items_per_thread, max_items_per_thread>,
                    sweep_config>(benchmarks, stream, size);
            }
        };

        void operator()(std::vector<benchmark::internal::Benchmark*>& benchmarks,
                        hipStream_t                                   stream,
                        size_t                                        size)
        {
            static_for_each<make_index_range<unsigned int, 6, 10>, sweep_config_2d>(
                benchmarks, stream, size);
        }
    };
};

void add_sort_keys_benchmarks(std::vector<benchmark::internal::Benchmark*>& benchmarks,
                              hipStream_t                                   stream,
                              size_t                                        size)
{
    using merge_block_size_range = make_index_range<unsigned int, 6, 10>;

    static_for_each<merge_block_size_range, create_benchmarks<int>::sweep_config_3d>(
        benchmarks, stream, size);

    static_for_each<merge_block_size_range, create_benchmarks<long long>::sweep_config_3d>(
        benchmarks, stream, size);

    static_for_each<merge_block_size_range, create_benchmarks<int8_t>::sweep_config_3d>(
        benchmarks, stream, size);

    static_for_each<merge_block_size_range, create_benchmarks<uint8_t>::sweep_config_3d>(
        benchmarks, stream, size);
    
    static_for_each<merge_block_size_range, create_benchmarks<rocprim::half>::sweep_config_3d>(
        benchmarks, stream, size);

    static_for_each<merge_block_size_range, create_benchmarks<short>::sweep_config_3d>(
        benchmarks, stream, size);
}

void add_sort_pairs_benchmarks(std::vector<benchmark::internal::Benchmark*>& benchmarks,
                              hipStream_t                                   stream,
                              size_t                                        size)
{
    using custom_float2  = custom_type<float, float>;
    using custom_double2 = custom_type<double, double>;

    using custom_int2            = custom_type<int, int>;
    using custom_char_double     = custom_type<char, double>;
    using custom_longlong_double = custom_type<long long, double>;

    using merge_block_size_range = make_index_range<unsigned int, 6, 10>;

    static_for_each<merge_block_size_range, create_benchmarks<int, float>::sweep_config_3d>(
        benchmarks, stream, size);

    static_for_each<merge_block_size_range, create_benchmarks<long long, double>::sweep_config_3d>(
        benchmarks, stream, size);

    static_for_each<merge_block_size_range, create_benchmarks<int8_t, int8_t>::sweep_config_3d>(
        benchmarks, stream, size);

    static_for_each<merge_block_size_range, create_benchmarks<uint8_t, uint8_t>::sweep_config_3d>(
        benchmarks, stream, size);
    
    static_for_each<merge_block_size_range, create_benchmarks<rocprim::half, rocprim::half>::sweep_config_3d>(
        benchmarks, stream, size);

    static_for_each<merge_block_size_range, create_benchmarks<short, short>::sweep_config_3d>(
        benchmarks, stream, size);

    static_for_each<merge_block_size_range, create_benchmarks<int, custom_float2>::sweep_config_3d>(
        benchmarks, stream, size);

    static_for_each<merge_block_size_range, create_benchmarks<long long, custom_double2>::sweep_config_3d>(
        benchmarks, stream, size);

    static_for_each<merge_block_size_range, create_benchmarks<custom_double2, custom_double2>::sweep_config_3d>(
        benchmarks, stream, size);

    static_for_each<merge_block_size_range, create_benchmarks<custom_int2, custom_double2>::sweep_config_3d>(
        benchmarks, stream, size);

    static_for_each<merge_block_size_range, create_benchmarks<custom_int2, custom_char_double>::sweep_config_3d>(
        benchmarks, stream, size);

    static_for_each<merge_block_size_range, create_benchmarks<custom_int2, custom_longlong_double>::sweep_config_3d>(
        benchmarks, stream, size);
}

#else // BENCHMARK_CONFIG_TUNING

#define CREATE_SORT_KEYS_BENCHMARK(Key) \
benchmark::RegisterBenchmark( \
    (std::string("sort_keys") + "<" #Key ">").c_str(), \
    [=](benchmark::State& state) { run_sort_keys_benchmark<Key>(state, stream, size); } \
)

void add_sort_keys_benchmarks(std::vector<benchmark::internal::Benchmark*>& benchmarks,
                              hipStream_t stream,
                              size_t size)
{
    std::vector<benchmark::internal::Benchmark*> bs =
    {
        CREATE_SORT_KEYS_BENCHMARK(int),
        CREATE_SORT_KEYS_BENCHMARK(long long),

        CREATE_SORT_KEYS_BENCHMARK(int8_t),
        CREATE_SORT_KEYS_BENCHMARK(uint8_t),
        CREATE_SORT_KEYS_BENCHMARK(rocprim::half),
        CREATE_SORT_KEYS_BENCHMARK(short),
    };
    benchmarks.insert(benchmarks.end(), bs.begin(), bs.end());
}

#define CREATE_SORT_PAIRS_BENCHMARK(Key, Value) \
benchmark::RegisterBenchmark( \
    (std::string("sort_pairs") + "<" #Key ", " #Value ">").c_str(), \
    [=](benchmark::State& state) { run_sort_pairs_benchmark<Key, Value>(state, stream, size); } \
)

void add_sort_pairs_benchmarks(std::vector<benchmark::internal::Benchmark*>& benchmarks,
                               hipStream_t stream,
                               size_t size)
{
    using custom_float2 = custom_type<float, float>;
    using custom_double2 = custom_type<double, double>;

    using custom_int2            = custom_type<int, int>;
    using custom_char_double     = custom_type<char, double>;
    using custom_longlong_double = custom_type<long long, double>;

    std::vector<benchmark::internal::Benchmark*> bs =
    {
        CREATE_SORT_PAIRS_BENCHMARK(int, float),
        CREATE_SORT_PAIRS_BENCHMARK(long long, double),

        CREATE_SORT_PAIRS_BENCHMARK(int8_t, int8_t),
        CREATE_SORT_PAIRS_BENCHMARK(uint8_t, uint8_t),
        CREATE_SORT_PAIRS_BENCHMARK(rocprim::half, rocprim::half),
        CREATE_SORT_PAIRS_BENCHMARK(short, short),

        CREATE_SORT_PAIRS_BENCHMARK(int, custom_float2),
        CREATE_SORT_PAIRS_BENCHMARK(long long, custom_double2),
        CREATE_SORT_PAIRS_BENCHMARK(custom_double2, custom_double2),
        CREATE_SORT_PAIRS_BENCHMARK(custom_int2, custom_double2),
        CREATE_SORT_PAIRS_BENCHMARK(custom_int2, custom_char_double),
        CREATE_SORT_PAIRS_BENCHMARK(custom_int2, custom_longlong_double),
    };
    benchmarks.insert(benchmarks.end(), bs.begin(), bs.end());
}

#endif // BENCHMARK_CONFIG_TUNING

int main(int argc, char *argv[])
{
    cli::Parser parser(argc, argv);
    parser.set_optional<size_t>("size", "size", DEFAULT_N, "number of values");
    parser.set_optional<int>("trials", "trials", -1, "number of iterations");
    parser.run_and_exit_if_error();

    // Parse argv
    benchmark::Initialize(&argc, argv);
    const size_t size = parser.get<size_t>("size");
    const int trials = parser.get<int>("trials");

    // HIP
    hipStream_t stream = 0; // default
    hipDeviceProp_t devProp;
    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    HIP_CHECK(hipGetDeviceProperties(&devProp, device_id));
    std::cout << "[HIP] Device name: " << devProp.name << std::endl;

    // Add benchmarks
    std::vector<benchmark::internal::Benchmark*> benchmarks;
    add_sort_keys_benchmarks(benchmarks, stream, size);
    add_sort_pairs_benchmarks(benchmarks, stream, size);

    // Use manual timing
    for(auto& b : benchmarks)
    {
        b->UseManualTime();
        b->Unit(benchmark::kMillisecond);
    }

    // Force number of iterations
    if(trials > 0)
    {
        for(auto& b : benchmarks)
        {
            b->Iterations(trials);
        }
    }

    // Run benchmarks
    benchmark::RunSpecifiedBenchmarks();
    return 0;
}
