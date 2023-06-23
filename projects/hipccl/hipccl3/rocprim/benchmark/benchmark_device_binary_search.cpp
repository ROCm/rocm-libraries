// MIT License
//
// Copyright (c) 2019-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

// Google Benchmark
#include "benchmark/benchmark.h"
// CmdParser
#include "cmdparser.hpp"
#include "benchmark_utils.hpp"

// HIP API
#include <hip/hip_runtime.h>

// rocPRIM
#include <rocprim/rocprim.hpp>

#include "benchmark_device_binary_search.parallel.hpp"

#ifndef DEFAULT_N
const size_t DEFAULT_N = 1024 * 1024 * 32;
#endif

const unsigned int batch_size = 10;
const unsigned int warmup_size = 5;

template<class T, class AlgorithmSelectorTag>
void run_benchmark(benchmark::State& state,
                   hipStream_t       stream,
                   size_t            haystack_size,
                   size_t            needles_size,
                   bool              sorted_needles)
{
    using haystack_type = T;
    using needle_type = T;
    using output_type = size_t;
    using compare_op_type = typename std::conditional<std::is_same<needle_type, rocprim::half>::value, half_less, rocprim::less<needle_type>>::type;

    compare_op_type compare_op;
    // Generate data
    std::vector<haystack_type> haystack(haystack_size);
    std::iota(haystack.begin(), haystack.end(), 0);

    std::vector<needle_type> needles = get_random_data<needle_type>(
        needles_size, needle_type(0), needle_type(haystack_size)
    );
    if(sorted_needles)
    {
        std::sort(needles.begin(), needles.end(), compare_op);
    }

    haystack_type * d_haystack;
    needle_type * d_needles;
    output_type * d_output;
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_haystack), haystack_size * sizeof(haystack_type)));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_needles), needles_size * sizeof(needle_type)));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_output), needles_size * sizeof(output_type)));
    HIP_CHECK(
        hipMemcpy(
            d_haystack, haystack.data(),
            haystack_size * sizeof(haystack_type),
            hipMemcpyHostToDevice
        )
    );
    HIP_CHECK(
        hipMemcpy(
            d_needles, needles.data(),
            needles_size * sizeof(needle_type),
            hipMemcpyHostToDevice
        )
    );

    void * d_temporary_storage = nullptr;
    size_t temporary_storage_bytes;
    HIP_CHECK(dispatch_binary_search(AlgorithmSelectorTag{},
                                     d_temporary_storage,
                                     temporary_storage_bytes,
                                     d_haystack,
                                     d_needles,
                                     d_output,
                                     haystack_size,
                                     needles_size,
                                     compare_op,
                                     stream));

    HIP_CHECK(hipMalloc(&d_temporary_storage, temporary_storage_bytes));

    // Warm-up
    for(size_t i = 0; i < warmup_size; i++)
    {
        HIP_CHECK(dispatch_binary_search(AlgorithmSelectorTag{},
                                         d_temporary_storage,
                                         temporary_storage_bytes,
                                         d_haystack,
                                         d_needles,
                                         d_output,
                                         haystack_size,
                                         needles_size,
                                         compare_op,
                                         stream));
    }
    HIP_CHECK(hipDeviceSynchronize());

    // HIP events creation
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    for(auto _ : state)
    {
        // Record start event
        HIP_CHECK(hipEventRecord(start, stream));

        for(size_t i = 0; i < batch_size; i++)
        {
            HIP_CHECK(dispatch_binary_search(AlgorithmSelectorTag{},
                                             d_temporary_storage,
                                             temporary_storage_bytes,
                                             d_haystack,
                                             d_needles,
                                             d_output,
                                             haystack_size,
                                             needles_size,
                                             compare_op,
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

    state.SetBytesProcessed(state.iterations() * batch_size * needles_size * sizeof(needle_type));
    state.SetItemsProcessed(state.iterations() * batch_size * needles_size);

    HIP_CHECK(hipFree(d_temporary_storage));
    HIP_CHECK(hipFree(d_haystack));
    HIP_CHECK(hipFree(d_needles));
    HIP_CHECK(hipFree(d_output));
}

#define CREATE_BENCHMARK(T, K, SORTED, ALGO_TAG)                                                 \
    benchmark::RegisterBenchmark(                                                                \
        bench_naming::format_name(                                                               \
            "{lvl:device,algo:" + ALGO_TAG{}.name() + ",key_type:" #T ",subalgo:" #K "_percent_" \
            + std::string(SORTED ? "sorted" : "random") + "_needles,cfg:default_config}")        \
            .c_str(),                                                                            \
        [=](benchmark::State& state)                                                             \
        { run_benchmark<T, ALGO_TAG>(state, stream, size, size * K / 100, SORTED); })

#define BENCHMARK_ALGORITHMS(T, K, SORTED)                        \
    CREATE_BENCHMARK(T, K, SORTED, binary_search_subalgorithm),   \
        CREATE_BENCHMARK(T, K, SORTED, lower_bound_subalgorithm), \
        CREATE_BENCHMARK(T, K, SORTED, upper_bound_subalgorithm)

#define BENCHMARK_TYPE(type) \
    BENCHMARK_ALGORITHMS(type, 10, true), BENCHMARK_ALGORITHMS(type, 10, false)

int main(int argc, char *argv[])
{
    cli::Parser parser(argc, argv);
    parser.set_optional<size_t>("size", "size", DEFAULT_N, "number of values");
    parser.set_optional<int>("trials", "trials", -1, "number of iterations");
    parser.set_optional<std::string>("name_format",
                                     "name_format",
                                     "human",
                                     "either: json,human,txt");
#ifdef BENCHMARK_CONFIG_TUNING
    // optionally run an evenly split subset of benchmarks, when making multiple program invocations
    parser.set_optional<int>("parallel_instance",
                             "parallel_instance",
                             0,
                             "parallel instance index");
    parser.set_optional<int>("parallel_instances",
                             "parallel_instances",
                             1,
                             "total parallel instances");
#endif
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

    using custom_float2 = custom_type<float, float>;
    using custom_double2 = custom_type<double, double>;

    // Add benchmarks
    std::vector<benchmark::internal::Benchmark*> benchmarks;
#ifdef BENCHMARK_CONFIG_TUNING
    const int parallel_instance  = parser.get<int>("parallel_instance");
    const int parallel_instances = parser.get<int>("parallel_instances");
    config_autotune_register::register_benchmark_subset(benchmarks,
                                                        parallel_instance,
                                                        parallel_instances,
                                                        size,
                                                        stream);
#else // BENCHMARK_CONFIG_TUNING
    benchmarks = {BENCHMARK_TYPE(float),
                  BENCHMARK_TYPE(double),
                  BENCHMARK_TYPE(int8_t),
                  BENCHMARK_TYPE(uint8_t),
                  BENCHMARK_TYPE(rocprim::half),
                  BENCHMARK_TYPE(custom_float2),
                  BENCHMARK_TYPE(custom_double2)};
#endif // BENCHMARK_CONFIG_TUNING

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
