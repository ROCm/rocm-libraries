// MIT License
//
// Copyright (c) 2017-2023 Advanced Micro Devices, Inc. All rights reserved.
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
#include <locale>
#include <codecvt>
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
    size = (size * sizeof(int)) / sizeof(T);

    std::vector<T> input;
    std::vector<FlagType> flags = get_random_data01<FlagType>(size, true_probability);
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
    HIP_CHECK(rocprim::partition(
        nullptr,
        temp_storage_size_bytes,
        d_input,
        d_flags,
        d_output,
        d_selected_count_output,
        input.size(),
        stream
    ));
    HIP_CHECK(hipDeviceSynchronize());

    // allocate temporary storage
    void * d_temp_storage = nullptr;
    HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_size_bytes));
    HIP_CHECK(hipDeviceSynchronize());

    // Warm-up
    for(size_t i = 0; i < 10; i++)
    {
        HIP_CHECK(rocprim::partition(
            d_temp_storage,
            temp_storage_size_bytes,
            d_input,
            d_flags,
            d_output,
            d_selected_count_output,
            input.size(),
            stream
        ));
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
            HIP_CHECK(rocprim::partition(
                d_temp_storage,
                temp_storage_size_bytes,
                d_input,
                d_flags,
                d_output,
                d_selected_count_output,
                input.size(),
                stream
            ));
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
}

template<class T>
void run_if_benchmark(benchmark::State& state,
                      size_t size,
                      const hipStream_t stream,
                      float true_probability)
{
    auto select_op = [true_probability] __device__ (const T& value) -> bool
    {
        if(value < T(127 * true_probability)) return true;
        return false;
    };

    std::vector<T> input = get_random_data<T>(size, T(0), T(127));
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
    HIP_CHECK(rocprim::partition(
        nullptr,
        temp_storage_size_bytes,
        d_input,
        d_output,
        d_selected_count_output,
        input.size(),
        select_op,
        stream
    ));
    HIP_CHECK(hipDeviceSynchronize());

    // allocate temporary storage
    void * d_temp_storage = nullptr;
    HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_size_bytes));
    HIP_CHECK(hipDeviceSynchronize());

    // Warm-up
    for(size_t i = 0; i < 10; i++)
    {
        HIP_CHECK(rocprim::partition(
            d_temp_storage,
            temp_storage_size_bytes,
            d_input,
            d_output,
            d_selected_count_output,
            input.size(),
            select_op,
            stream
        ));
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
            HIP_CHECK(rocprim::partition(
                d_temp_storage,
                temp_storage_size_bytes,
                d_input,
                d_output,
                d_selected_count_output,
                input.size(),
                select_op,
                stream
            ));
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

template<class T>
void run_two_way_benchmark(benchmark::State& state,
                           size_t            size,
                           const hipStream_t stream,
                           float             probability)
{
    auto predicate
        = [probability] __device__(const T& value) { return value < T(127 * probability); };

    std::vector<T> input = get_random_data<T>(size, T(0), T(127));
    T*             d_input;
    T*             d_output_selected;
    T*             d_output_rejected;
    unsigned int*  d_selected_count_output;
    HIP_CHECK(hipMalloc(&d_input, input.size() * sizeof(T)));
    HIP_CHECK(hipMalloc(&d_output_selected, input.size() * sizeof(T)));
    HIP_CHECK(hipMalloc(&d_output_rejected, input.size() * sizeof(T)));
    HIP_CHECK(hipMalloc(&d_selected_count_output, sizeof(*d_selected_count_output)));
    HIP_CHECK(hipMemcpy(d_input, input.data(), input.size() * sizeof(T), hipMemcpyHostToDevice));
    HIP_CHECK(hipDeviceSynchronize());
    // Allocate temporary storage memory
    size_t temp_storage_size_bytes;

    // Get size of d_temp_storage
    HIP_CHECK(rocprim::partition_two_way(nullptr,
                                         temp_storage_size_bytes,
                                         d_input,
                                         d_output_selected,
                                         d_output_rejected,
                                         d_selected_count_output,
                                         input.size(),
                                         predicate,
                                         stream));

    // allocate temporary storage
    void* d_temp_storage = nullptr;
    HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_size_bytes));

    // Warm-up
    for(size_t i = 0; i < 10; i++)
    {
        HIP_CHECK(rocprim::partition_two_way(d_temp_storage,
                                             temp_storage_size_bytes,
                                             d_input,
                                             d_output_selected,
                                             d_output_rejected,
                                             d_selected_count_output,
                                             input.size(),
                                             predicate,
                                             stream));
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
            HIP_CHECK(rocprim::partition_two_way(d_temp_storage,
                                                 temp_storage_size_bytes,
                                                 d_input,
                                                 d_output_selected,
                                                 d_output_rejected,
                                                 d_selected_count_output,
                                                 input.size(),
                                                 predicate,
                                                 stream));
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
    hipFree(d_output_selected);
    hipFree(d_output_rejected);
    hipFree(d_selected_count_output);
    hipFree(d_temp_storage);
}

template<class T>
void run_three_way_benchmark(benchmark::State& state,
                             size_t size,
                             const hipStream_t stream,
                             float first_probability,
                             float second_probability)
{
    auto first_select_op = [first_probability] __device__ (const T& value)
    {
        return value < T(127 * first_probability);
    };
    auto second_select_op = [second_probability] __device__ (const T& value)
    {
        return value < T(127 * second_probability);
    };

    std::vector<T> input = get_random_data<T>(size, T(0), T(127));
    T * d_input;
    T * d_output_first;
    T * d_output_second;
    T * d_output_unselected;
    unsigned int * d_selected_count_output;
    HIP_CHECK(hipMalloc(&d_input, input.size() * sizeof(T)));
    HIP_CHECK(hipMalloc(&d_output_first, input.size() * sizeof(T)));
    HIP_CHECK(hipMalloc(&d_output_second, input.size() * sizeof(T)));
    HIP_CHECK(hipMalloc(&d_output_unselected, input.size() * sizeof(T)));
    HIP_CHECK(hipMalloc(&d_selected_count_output, 2 * sizeof(unsigned int)));
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
    HIP_CHECK(rocprim::partition_three_way(
        nullptr,
        temp_storage_size_bytes,
        d_input,
        d_output_first,
        d_output_second,
        d_output_unselected,
        d_selected_count_output,
        input.size(),
        first_select_op,
        second_select_op,
        stream
    ));
    HIP_CHECK(hipDeviceSynchronize());

    // allocate temporary storage
    void * d_temp_storage = nullptr;
    HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_size_bytes));
    HIP_CHECK(hipDeviceSynchronize());

    // Warm-up
    for(size_t i = 0; i < 10; i++)
    {
        HIP_CHECK(rocprim::partition_three_way(
            d_temp_storage,
            temp_storage_size_bytes,
            d_input,
            d_output_first,
            d_output_second,
            d_output_unselected,
            d_selected_count_output,
            input.size(),
            first_select_op,
            second_select_op,
            stream
        ));
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
            HIP_CHECK(rocprim::partition_three_way(
                d_temp_storage,
                temp_storage_size_bytes,
                d_input,
                d_output_first,
                d_output_second,
                d_output_unselected,
                d_selected_count_output,
                input.size(),
                first_select_op,
                second_select_op,
                stream
            ));
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
    hipFree(d_output_first);
    hipFree(d_output_second);
    hipFree(d_output_unselected);
    hipFree(d_selected_count_output);
    hipFree(d_temp_storage);
}

#define CREATE_PARTITION_FLAGGED_BENCHMARK(T, F, p)                                 \
    benchmark::RegisterBenchmark(                                                   \
        bench_naming::format_name("{lvl:device,algo:partition,key_type:" #T         \
                                  ",subalgo:flags,flag_type:" #F ",probability:" #p \
                                  ",cfg:default_config}")                           \
            .c_str(),                                                               \
        run_flagged_benchmark<T, F>,                                                \
        size,                                                                       \
        stream,                                                                     \
        p)

#define CREATE_PARTITION_IF_BENCHMARK(T, p)                                             \
    benchmark::RegisterBenchmark(                                                       \
        bench_naming::format_name("{lvl:device,algo:partition,key_type:" #T             \
                                  ",subalgo:if,probability:" #p ",cfg:default_config}") \
            .c_str(),                                                                   \
        run_if_benchmark<T>,                                                            \
        size,                                                                           \
        stream,                                                                         \
        p)

#define CREATE_PARTITION_TWO_WAY_BENCHMARK(T, p)                                             \
    benchmark::RegisterBenchmark(                                                            \
        bench_naming::format_name("{lvl:device,algo:partition,key_type:" #T                  \
                                  ",subalgo:two_way,probability:" #p ",cfg:default_config}") \
            .c_str(),                                                                        \
        run_two_way_benchmark<T>,                                                            \
        size,                                                                                \
        stream,                                                                              \
        p)

#define CREATE_PARTITION_THREE_WAY_BENCHMARK(T, p1, p2)                                       \
    benchmark::RegisterBenchmark(                                                             \
        bench_naming::format_name("{lvl:device,algo:partition,key_type:" #T                   \
                                  ",subalgo:three_way,probability1:" #p1 ",probability2:" #p2 \
                                  ",cfg:default_config}")                                     \
            .c_str(),                                                                         \
        run_three_way_benchmark<T>,                                                           \
        size,                                                                                 \
        stream,                                                                               \
        p1,                                                                                   \
        p2)

// clang-format off

#define BENCHMARK_FLAGGED_TYPE(type, value) \
    CREATE_PARTITION_FLAGGED_BENCHMARK(type, value, 0.05f), \
    CREATE_PARTITION_FLAGGED_BENCHMARK(type, value, 0.25f), \
    CREATE_PARTITION_FLAGGED_BENCHMARK(type, value, 0.5f), \
    CREATE_PARTITION_FLAGGED_BENCHMARK(type, value, 0.75f)

#define BENCHMARK_IF_TYPE(type) \
    CREATE_PARTITION_IF_BENCHMARK(type, 0.05f), \
    CREATE_PARTITION_IF_BENCHMARK(type, 0.25f), \
    CREATE_PARTITION_IF_BENCHMARK(type, 0.5f), \
    CREATE_PARTITION_IF_BENCHMARK(type, 0.75f)

#define BENCHMARK_TWO_WAY_TYPE(type) \
    CREATE_PARTITION_TWO_WAY_BENCHMARK(type, 0.05f), \
    CREATE_PARTITION_TWO_WAY_BENCHMARK(type, 0.25f), \
    CREATE_PARTITION_TWO_WAY_BENCHMARK(type, 0.5f), \
    CREATE_PARTITION_TWO_WAY_BENCHMARK(type, 0.75f)

#define BENCHMARK_THREE_WAY_TYPE(type) \
    CREATE_PARTITION_THREE_WAY_BENCHMARK(type, 0.05f, 0.25f), \
    CREATE_PARTITION_THREE_WAY_BENCHMARK(type, 0.25f, 0.5f), \
    CREATE_PARTITION_THREE_WAY_BENCHMARK(type, 0.5f, 0.75f), \
    CREATE_PARTITION_THREE_WAY_BENCHMARK(type, 0.75f, 1.f)

// clang-format on

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
    std::vector<benchmark::internal::Benchmark*> benchmarks
        = {BENCHMARK_FLAGGED_TYPE(int, unsigned char),
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

           BENCHMARK_TWO_WAY_TYPE(int),
           BENCHMARK_TWO_WAY_TYPE(float),
           BENCHMARK_TWO_WAY_TYPE(double),
           BENCHMARK_TWO_WAY_TYPE(uint8_t),
           BENCHMARK_TWO_WAY_TYPE(int8_t),
           BENCHMARK_TWO_WAY_TYPE(rocprim::half),
           BENCHMARK_TWO_WAY_TYPE(custom_int_double),

           BENCHMARK_THREE_WAY_TYPE(int),
           BENCHMARK_THREE_WAY_TYPE(float),
           BENCHMARK_THREE_WAY_TYPE(double),
           BENCHMARK_THREE_WAY_TYPE(uint8_t),
           BENCHMARK_THREE_WAY_TYPE(int8_t),
           BENCHMARK_THREE_WAY_TYPE(rocprim::half),
           BENCHMARK_THREE_WAY_TYPE(custom_int_double)};

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
