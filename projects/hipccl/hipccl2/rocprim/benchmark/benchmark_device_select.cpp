// MIT License
//
// Copyright (c) 2017-2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include <codecvt>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <locale>
#include <string>
#include <vector>

// Google Benchmark
#include "benchmark/benchmark.h"
// CmdParser
#include "cmdparser.hpp"
#include "benchmark_utils.hpp"

// HIP API
#include <hip/hip_runtime.h>

#include <rocprim/rocprim.hpp>

#ifndef DEFAULT_N
const size_t DEFAULT_N = 1024 * 1024 * 32;
#endif

template<class T, class FlagType>
void run_flagged_benchmark(benchmark::State& state,
                           size_t size,
                           const hipStream_t stream,
                           float true_probability)
{
    std::vector<T> input;
    std::vector<FlagType> flags = get_random_data01<FlagType>(size, true_probability);
    std::vector<unsigned int> selected_count_output(1);
    if(std::is_floating_point<T>::value)
    {
        input = get_random_data<T>(size, T(-1000), T(1000));
    }
    else
    {
        input = get_random_data<T>(
            size,
            std::numeric_limits<T>::min(),
            std::numeric_limits<T>::max()
        );
    }

    T * d_input;
    FlagType * d_flags;
    T * d_output;
    unsigned int * d_selected_count_output;
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_input), input.size() * sizeof(T)));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_flags), flags.size() * sizeof(FlagType)));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_output), input.size() * sizeof(T)));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_selected_count_output), sizeof(unsigned int)));
    HIP_CHECK(
        hipMemcpy(
            d_input, input.data(),
            input.size() * sizeof(T),
            hipMemcpyHostToDevice
        )
    );
    HIP_CHECK(
        hipMemcpy(
            d_flags, flags.data(),
            flags.size() * sizeof(FlagType),
            hipMemcpyHostToDevice
        )
    );
    HIP_CHECK(hipDeviceSynchronize());
    // Allocate temporary storage memory
    size_t temp_storage_size_bytes;

    // Get size of d_temp_storage
    rocprim::select(
        nullptr,
        temp_storage_size_bytes,
        d_input,
        d_flags,
        d_output,
        d_selected_count_output,
        input.size(),
        stream
    );
    HIP_CHECK(hipDeviceSynchronize());

    // allocate temporary storage
    void * d_temp_storage = nullptr;
    HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_size_bytes));
    HIP_CHECK(hipDeviceSynchronize());

    // Warm-up
    for(size_t i = 0; i < 10; i++)
    {
        rocprim::select(
            d_temp_storage,
            temp_storage_size_bytes,
            d_input,
            d_flags,
            d_output,
            d_selected_count_output,
            input.size(),
            stream
        );
    }
    HIP_CHECK(hipDeviceSynchronize());

    // HIP events creation
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    const unsigned int batch_size = 10;
    for(auto _ : state)
    {
        // Record start event
        HIP_CHECK(hipEventRecord(start, stream));

        for(size_t i = 0; i < batch_size; i++)
        {
            rocprim::select(
                d_temp_storage,
                temp_storage_size_bytes,
                d_input,
                d_flags,
                d_output,
                d_selected_count_output,
                input.size(),
                stream
            );
        }

        // Record stop event and wait until it completes
        HIP_CHECK(hipEventRecord(stop, stream));
        HIP_CHECK(hipEventSynchronize(stop));

        float elapsed_mseconds;
        HIP_CHECK(hipEventElapsedTime(&elapsed_mseconds, start, stop));
        state.SetIterationTime(elapsed_mseconds / 1000);
    }

    // Destroy HIP events
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));

    state.SetBytesProcessed(state.iterations() * batch_size * size * sizeof(T));
    state.SetItemsProcessed(state.iterations() * batch_size * size);

    hipFree(d_input);
    hipFree(d_flags);
    hipFree(d_output);
    hipFree(d_selected_count_output);
    hipFree(d_temp_storage);
    HIP_CHECK(hipDeviceSynchronize());
}

template<class T>
void run_selectop_benchmark(benchmark::State& state,
                            size_t size,
                            const hipStream_t stream,
                            float true_probability)
{
    std::vector<T> input = get_random_data<T>(size, T(0), T(1000));
    std::vector<unsigned int> selected_count_output(1);

    auto select_op = [true_probability] __device__ (const T& value) -> bool
    {
        if(value < T(1000 * true_probability)) return true;
        return false;
    };

    T * d_input;
    T * d_output;
    unsigned int * d_selected_count_output;
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_input), input.size() * sizeof(T)));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_output), input.size() * sizeof(T)));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_selected_count_output), sizeof(unsigned int)));
    HIP_CHECK(
        hipMemcpy(
            d_input, input.data(),
            input.size() * sizeof(T),
            hipMemcpyHostToDevice
        )
    );
    HIP_CHECK(hipDeviceSynchronize());

    // Allocate temporary storage memory
    size_t temp_storage_size_bytes;

    // Get size of d_temp_storage
    rocprim::select(
        nullptr,
        temp_storage_size_bytes,
        d_input,
        d_output,
        d_selected_count_output,
        input.size(),
        select_op,
        stream
    );
    HIP_CHECK(hipDeviceSynchronize());

    // allocate temporary storage
    void * d_temp_storage = nullptr;
    HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_size_bytes));
    HIP_CHECK(hipDeviceSynchronize());

    // Warm-up
    for(size_t i = 0; i < 10; i++)
    {
        rocprim::select(
            d_temp_storage,
            temp_storage_size_bytes,
            d_input,
            d_output,
            d_selected_count_output,
            input.size(),
            select_op,
            stream
        );
    }
    HIP_CHECK(hipDeviceSynchronize());

    // HIP events creation
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    const unsigned int batch_size = 10;
    for(auto _ : state)
    {
        // Record start event
        HIP_CHECK(hipEventRecord(start, stream));

        for(size_t i = 0; i < batch_size; i++)
        {
            rocprim::select(
                d_temp_storage,
                temp_storage_size_bytes,
                d_input,
                d_output,
                d_selected_count_output,
                input.size(),
                select_op,
                stream
            );
        }

        // Record stop event and wait until it completes
        HIP_CHECK(hipEventRecord(stop, stream));
        HIP_CHECK(hipEventSynchronize(stop));

        float elapsed_mseconds;
        HIP_CHECK(hipEventElapsedTime(&elapsed_mseconds, start, stop));
        state.SetIterationTime(elapsed_mseconds / 1000);
    }

    // Destroy HIP events
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));

    state.SetBytesProcessed(state.iterations() * batch_size * size * sizeof(T));
    state.SetItemsProcessed(state.iterations() * batch_size * size);

    hipFree(d_input);
    hipFree(d_output);
    hipFree(d_selected_count_output);
    hipFree(d_temp_storage);
    HIP_CHECK(hipDeviceSynchronize());
}

template<class T>
void run_unique_benchmark(benchmark::State& state,
                          size_t size,
                          const hipStream_t stream,
                          float discontinuity_probability)
{
    using op_type = typename std::conditional<std::is_same<T, rocprim::half>::value, half_plus, rocprim::plus<T>>::type;
    op_type op;

    std::vector<T> input(size);
    {
        auto input01 = get_random_data01<T>(size, discontinuity_probability);
        auto acc = input01[0];
        input[0] = acc;
        for(size_t i = 1; i < input01.size(); i++)
        {
            input[i] = op(acc, input01[i]);
        }
    }
    std::vector<unsigned int> selected_count_output(1);
    auto equality_op = rocprim::equal_to<T>();

    T * d_input;
    T * d_output;
    unsigned int * d_selected_count_output;
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_input), input.size() * sizeof(T)));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_output), input.size() * sizeof(T)));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_selected_count_output), sizeof(unsigned int)));
    HIP_CHECK(
        hipMemcpy(
            d_input, input.data(),
            input.size() * sizeof(T),
            hipMemcpyHostToDevice
        )
    );
    HIP_CHECK(hipDeviceSynchronize());

    // Allocate temporary storage memory
    size_t temp_storage_size_bytes;

    // Get size of d_temp_storage
    rocprim::unique(
        nullptr,
        temp_storage_size_bytes,
        d_input,
        d_output,
        d_selected_count_output,
        input.size(),
        equality_op,
        stream
    );
    HIP_CHECK(hipDeviceSynchronize());

    // allocate temporary storage
    void * d_temp_storage = nullptr;
    HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_size_bytes));
    HIP_CHECK(hipDeviceSynchronize());

    // Warm-up
    for(size_t i = 0; i < 10; i++)
    {
        rocprim::unique(
            d_temp_storage,
            temp_storage_size_bytes,
            d_input,
            d_output,
            d_selected_count_output,
            input.size(),
            equality_op,
            stream
        );
    }
    HIP_CHECK(hipDeviceSynchronize());

    // HIP events creation
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    const unsigned int batch_size = 10;
    for(auto _ : state)
    {
        // Record start event
        HIP_CHECK(hipEventRecord(start, stream));

        for(size_t i = 0; i < batch_size; i++)
        {
            rocprim::unique(
                d_temp_storage,
                temp_storage_size_bytes,
                d_input,
                d_output,
                d_selected_count_output,
                input.size(),
                equality_op,
                stream
            );
        }

        // Record stop event and wait until it completes
        HIP_CHECK(hipEventRecord(stop, stream));
        HIP_CHECK(hipEventSynchronize(stop));

        float elapsed_mseconds;
        HIP_CHECK(hipEventElapsedTime(&elapsed_mseconds, start, stop));
        state.SetIterationTime(elapsed_mseconds / 1000);
    }

    // Destroy HIP events
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));

    state.SetBytesProcessed(state.iterations() * batch_size * size * sizeof(T));
    state.SetItemsProcessed(state.iterations() * batch_size * size);

    hipFree(d_input);
    hipFree(d_output);
    hipFree(d_selected_count_output);
    hipFree(d_temp_storage);
}

template <typename Key, typename Value>
void run_unique_by_key_benchmark(benchmark::State& state,
                                 size_t            size,
                                 const hipStream_t stream,
                                 float             discontinuity_probability)
{
    using op_type = typename std::
        conditional_t<std::is_same<Key, rocprim::half>::value, half_plus, rocprim::plus<Key>>;
    op_type op;

    std::vector<Key> input_keys(size);
    {
        auto input01  = get_random_data01<Key>(size, discontinuity_probability);
        auto acc      = input01[0];
        input_keys[0] = acc;
        for(size_t i = 1; i < input01.size(); i++)
        {
            input_keys[i] = op(acc, input01[i]);
        }
    }
    const auto                input_values = get_random_data<Value>(size, -1000, 1000);
    std::vector<unsigned int> selected_count_output(1);
    auto                      equality_op = rocprim::equal_to<Key>();

    Key*          d_keys_input;
    Value*        d_values_input;
    Key*          d_keys_output;
    Value*        d_values_output;
    unsigned int* d_selected_count_output;
    HIP_CHECK(hipMalloc(&d_keys_input, input_keys.size() * sizeof(input_keys[0])));
    HIP_CHECK(hipMalloc(&d_values_input, input_values.size() * sizeof(input_values[0])));
    HIP_CHECK(hipMalloc(&d_keys_output, input_keys.size() * sizeof(input_keys[0])));
    HIP_CHECK(hipMalloc(&d_values_output, input_values.size() * sizeof(input_values[0])));
    HIP_CHECK(hipMalloc(&d_selected_count_output, sizeof(selected_count_output[0])));
    HIP_CHECK(hipMemcpy(d_keys_input,
                        input_keys.data(),
                        input_keys.size() * sizeof(input_keys[0]),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_values_input,
                        input_values.data(),
                        input_values.size() * sizeof(input_values[0]),
                        hipMemcpyHostToDevice));

    // Allocate temporary storage memory
    size_t temp_storage_size_bytes;

    // Get size of d_temp_storage
    rocprim::unique_by_key(nullptr,
                           temp_storage_size_bytes,
                           d_keys_input,
                           d_values_input,
                           d_keys_output,
                           d_values_output,
                           d_selected_count_output,
                           input_keys.size(),
                           equality_op,
                           stream);
    HIP_CHECK(hipDeviceSynchronize());

    // allocate temporary storage
    void* d_temp_storage = nullptr;
    HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_size_bytes));
    HIP_CHECK(hipDeviceSynchronize());

    // Warm-up
    for(size_t i = 0; i < 10; i++)
    {
        rocprim::unique_by_key(d_temp_storage,
                               temp_storage_size_bytes,
                               d_keys_input,
                               d_values_input,
                               d_keys_output,
                               d_values_output,
                               d_selected_count_output,
                               input_keys.size(),
                               equality_op,
                               stream);
    }
    HIP_CHECK(hipDeviceSynchronize());

    // HIP events creation
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    const unsigned int batch_size = 10;
    for(auto _ : state)
    {
        // Record start event
        HIP_CHECK(hipEventRecord(start, stream));

        for(size_t i = 0; i < batch_size; i++)
        {
            rocprim::unique_by_key(d_temp_storage,
                                   temp_storage_size_bytes,
                                   d_keys_input,
                                   d_values_input,
                                   d_keys_output,
                                   d_values_output,
                                   d_selected_count_output,
                                   input_keys.size(),
                                   equality_op,
                                   stream);
        }

        // Record stop event and wait until it completes
        HIP_CHECK(hipEventRecord(stop, stream));
        HIP_CHECK(hipEventSynchronize(stop));

        float elapsed_mseconds;
        HIP_CHECK(hipEventElapsedTime(&elapsed_mseconds, start, stop));
        state.SetIterationTime(elapsed_mseconds / 1000);
    }

    // Destroy HIP events
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));

    state.SetBytesProcessed(state.iterations() * batch_size * size * (sizeof(Key) + sizeof(Value)));
    state.SetItemsProcessed(state.iterations() * batch_size * size);

    hipFree(d_keys_input);
    hipFree(d_values_input);
    hipFree(d_keys_output);
    hipFree(d_values_output);
    hipFree(d_selected_count_output);
    hipFree(d_temp_storage);
}

#define CREATE_SELECT_FLAGGED_BENCHMARK(T, F, p)                                              \
    benchmark::RegisterBenchmark(                                                             \
        bench_naming::format_name("{lvl:device,algo:select,subalgo:flagged,key_type:" #T      \
                                  ",flag_type:" #F ",probability:" #p ",cfg:default_config}") \
            .c_str(),                                                                         \
        run_flagged_benchmark<T, F>,                                                          \
        size,                                                                                 \
        stream,                                                                               \
        p)

#define CREATE_SELECT_IF_BENCHMARK(T, p)                                            \
    benchmark::RegisterBenchmark(                                                   \
        bench_naming::format_name("{lvl:device,algo:select,subalgo:if,key_type:" #T \
                                  ",probability:" #p ",cfg:default_config}")        \
            .c_str(),                                                               \
        run_selectop_benchmark<T>,                                                  \
        size,                                                                       \
        stream,                                                                     \
        p)

#define CREATE_UNIQUE_BENCHMARK(T, p)                                                   \
    benchmark::RegisterBenchmark(                                                       \
        bench_naming::format_name("{lvl:device,algo:unique,key_type:" #T ",value_type:" \
                                  + std::string(Traits<rocprim::empty_type>::name())    \
                                  + ",probability:" #p ",cfg:default_config}")          \
            .c_str(),                                                                   \
        run_unique_benchmark<T>,                                                        \
        size,                                                                           \
        stream,                                                                         \
        p)

#define CREATE_UNIQUE_BY_KEY_BENCHMARK(K, V, p)                                                   \
    benchmark::RegisterBenchmark(bench_naming::format_name("{lvl:device,algo:unique,key_type:" #K \
                                                           ",value_type:" #V ",probability:" #p   \
                                                           ",cfg:default_config}")                \
                                     .c_str(),                                                    \
                                 run_unique_by_key_benchmark<K, V>,                               \
                                 size,                                                            \
                                 stream,                                                          \
                                 p)

#define BENCHMARK_FLAGGED_TYPE(type, value) \
    CREATE_SELECT_FLAGGED_BENCHMARK(type, value, 0.05f), \
    CREATE_SELECT_FLAGGED_BENCHMARK(type, value, 0.25f), \
    CREATE_SELECT_FLAGGED_BENCHMARK(type, value, 0.5f), \
    CREATE_SELECT_FLAGGED_BENCHMARK(type, value, 0.75f)

#define BENCHMARK_IF_TYPE(type) \
    CREATE_SELECT_IF_BENCHMARK(type, 0.05f), \
    CREATE_SELECT_IF_BENCHMARK(type, 0.25f), \
    CREATE_SELECT_IF_BENCHMARK(type, 0.5f), \
    CREATE_SELECT_IF_BENCHMARK(type, 0.75f)

#define BENCHMARK_UNIQUE_TYPE(type) \
    CREATE_UNIQUE_BENCHMARK(type, 0.05f), \
    CREATE_UNIQUE_BENCHMARK(type, 0.25f), \
    CREATE_UNIQUE_BENCHMARK(type, 0.5f), \
    CREATE_UNIQUE_BENCHMARK(type, 0.75f)

#define BENCHMARK_UNIQUE_BY_KEY_TYPE(K, V) \
    CREATE_UNIQUE_BY_KEY_BENCHMARK(K, V, 0.05f), \
    CREATE_UNIQUE_BY_KEY_BENCHMARK(K, V, 0.25f), \
    CREATE_UNIQUE_BY_KEY_BENCHMARK(K, V, 0.5f), \
    CREATE_UNIQUE_BY_KEY_BENCHMARK(K, V, 0.75f)

int main(int argc, char *argv[])
{
    cli::Parser parser(argc, argv);
    parser.set_optional<size_t>("size", "size", DEFAULT_N, "number of values");
    parser.set_optional<int>("trials", "trials", -1, "number of iterations");
    parser.set_optional<std::string>("name_format",
                                     "name_format",
                                     "human",
                                     "either: json,human,txt");
    parser.run_and_exit_if_error();

    // Parse argv
    benchmark::Initialize(&argc, argv);
    const size_t size = parser.get<size_t>("size");
    const int trials = parser.get<int>("trials");
    bench_naming::set_format(parser.get<std::string>("name_format"));

    // HIP
    hipStream_t stream = 0; // default

    // Benchmark info
    add_common_benchmark_info();
    benchmark::AddCustomContext("size", std::to_string(size));

    using custom_double2 = custom_type<double, double>;
    using custom_int_double = custom_type<int, double>;

    // Add benchmarks
    std::vector<benchmark::internal::Benchmark*> benchmarks =
    {
        BENCHMARK_FLAGGED_TYPE(int, unsigned char),
        BENCHMARK_FLAGGED_TYPE(float, unsigned char),
        BENCHMARK_FLAGGED_TYPE(double, unsigned char),
        BENCHMARK_FLAGGED_TYPE(uint8_t, uint8_t),
        BENCHMARK_FLAGGED_TYPE(int8_t, int8_t),
        BENCHMARK_FLAGGED_TYPE(rocprim::half, int8_t),
        BENCHMARK_FLAGGED_TYPE(custom_double2, unsigned char),

        BENCHMARK_IF_TYPE(int),
        BENCHMARK_IF_TYPE(float),
        BENCHMARK_IF_TYPE(double),
        BENCHMARK_IF_TYPE(uint8_t),
        BENCHMARK_IF_TYPE(int8_t),
        BENCHMARK_IF_TYPE(rocprim::half),
        BENCHMARK_IF_TYPE(custom_int_double),

        BENCHMARK_UNIQUE_TYPE(int),
        BENCHMARK_UNIQUE_TYPE(float),
        BENCHMARK_UNIQUE_TYPE(double),
        BENCHMARK_UNIQUE_TYPE(uint8_t),
        BENCHMARK_UNIQUE_TYPE(int8_t),
        BENCHMARK_UNIQUE_TYPE(rocprim::half),
        BENCHMARK_UNIQUE_TYPE(custom_int_double),

        BENCHMARK_UNIQUE_BY_KEY_TYPE(int, int),
        BENCHMARK_UNIQUE_BY_KEY_TYPE(float, double),
        BENCHMARK_UNIQUE_BY_KEY_TYPE(double, custom_double2),
        BENCHMARK_UNIQUE_BY_KEY_TYPE(uint8_t, uint8_t),
        BENCHMARK_UNIQUE_BY_KEY_TYPE(int8_t, double),
        BENCHMARK_UNIQUE_BY_KEY_TYPE(rocprim::half, rocprim::half),
        BENCHMARK_UNIQUE_BY_KEY_TYPE(custom_int_double, custom_int_double)
    };

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
