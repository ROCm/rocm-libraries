// MIT License
//
// Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "benchmark_utils.hpp"
#include "cmdparser.hpp"

#include <benchmark/benchmark.h>

#include <hip/hip_runtime.h>

// rocPRIM
#include <rocprim/device/device_transform.hpp>
#include <rocprim/iterator/predicate_iterator.hpp>
#include <rocprim/iterator/transform_iterator.hpp>

#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <cstdio>
#include <cstdlib>

#ifndef DEFAULT_BYTES
const size_t DEFAULT_BYTES = 1024 * 1024 * 128 * 4;
#endif

const unsigned int batch_size  = 10;
const unsigned int warmup_size = 5;

template<class T>
struct identity
{
    __device__ T operator()(T value)
    {
        return value;
    }
};

template<class T, int C>
struct less_than
{
    __device__ bool operator()(T value) const
    {
        return value < T{C};
    }
};

template<class T, int I>
struct increment
{
    __device__ T operator()(T value) const
    {
        return value + T{I};
    }
};

template<class T, class Predicate, class Transform>
struct transform_op
{
    __device__
    auto operator()(T v) const
    {
        return Predicate{}(v) ? Transform{}(v) : v;
    }
};

template<class T, class Predicate, class Transform>
struct transform_it
{
    using value_type = T;

    void operator()(T* d_input, T* d_output, const size_t size, const hipStream_t stream)
    {
        auto t_it
            = rocprim::make_transform_iterator(d_input, transform_op<T, Predicate, Transform>{});
        HIP_CHECK(rocprim::transform(t_it, d_output, size, identity<T>{}, stream));
    }
};

template<class T, class Predicate, class Transform>
struct read_predicate_it
{
    using value_type = T;

    void operator()(T* d_input, T* d_output, const size_t size, const hipStream_t stream)
    {
        auto t_it = rocprim::make_transform_iterator(d_input, Transform{});
        auto r_it = rocprim::make_predicate_iterator(t_it, d_input, Predicate{});
        HIP_CHECK(rocprim::transform(r_it, d_output, size, identity<T>{}, stream));
    }
};

template<class T, class Predicate, class Transform>
struct write_predicate_it
{
    using value_type = T;

    void operator()(T* d_input, T* d_output, const size_t size, const hipStream_t stream)
    {
        auto t_it = rocprim::make_transform_iterator(d_input, Transform{});
        auto w_it = rocprim::make_predicate_iterator(d_output, d_input, Predicate{});
        HIP_CHECK(rocprim::transform(t_it, w_it, size, identity<T>{}, stream));
    }
};

template<class IteratorBenchmark>
void run_benchmark(benchmark::State&   state,
                   size_t              bytes,
                   const managed_seed& seed,
                   hipStream_t         stream)
{
    using T = typename IteratorBenchmark::value_type;

    // Calculate the number of elements 
    size_t size = bytes / sizeof(T);

    const auto     random_range = limit_random_range<T>(0, 99);
    std::vector<T> input
        = get_random_data<T>(size, random_range.first, random_range.second, seed.get_0());
    T*             d_input;
    T*             d_output;
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_input), size * sizeof(T)));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_output), size * sizeof(T)));
    HIP_CHECK(hipMemcpy(d_input, input.data(), size * sizeof(T), hipMemcpyHostToDevice));
    HIP_CHECK(hipDeviceSynchronize());

    // Warm-up
    for(size_t i = 0; i < warmup_size; i++)
    {
        IteratorBenchmark{}(d_input, d_output, size, stream);
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
            IteratorBenchmark{}(d_input, d_output, size, stream);
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

    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));
}

#define CREATE_BENCHMARK(B, T, C)                                                                  \
    benchmark::RegisterBenchmark(bench_naming::format_name("{lvl:device,algo:" #B ",p:p" #C        \
                                                           ",key_type:" #T ",cfg:default_config}") \
                                     .c_str(),                                                     \
                                 run_benchmark<B<T, less_than<T, C>, increment<T, 5>>>,            \
                                 bytes,                                                             \
                                 seed,                                                             \
                                 stream)

// clang-format off
#define CREATE_TYPED_BENCHMARK(T)                \
    CREATE_BENCHMARK(transform_it, T, 0),        \
    CREATE_BENCHMARK(read_predicate_it, T, 0),   \
    CREATE_BENCHMARK(write_predicate_it, T, 0),  \
    CREATE_BENCHMARK(transform_it, T, 25),       \
    CREATE_BENCHMARK(read_predicate_it, T, 25),  \
    CREATE_BENCHMARK(write_predicate_it, T, 25), \
    CREATE_BENCHMARK(transform_it, T, 50),       \
    CREATE_BENCHMARK(read_predicate_it, T, 50),  \
    CREATE_BENCHMARK(write_predicate_it, T, 50), \
    CREATE_BENCHMARK(transform_it, T, 75),       \
    CREATE_BENCHMARK(read_predicate_it, T, 75),  \
    CREATE_BENCHMARK(write_predicate_it, T, 75), \
    CREATE_BENCHMARK(transform_it, T, 100),      \
    CREATE_BENCHMARK(read_predicate_it, T, 100), \
    CREATE_BENCHMARK(write_predicate_it, T, 100)
// clang-format on

int main(int argc, char* argv[])
{
    cli::Parser parser(argc, argv);
    parser.set_optional<size_t>("size", "size", DEFAULT_BYTES, "number of bytes");
    parser.set_optional<int>("trials", "trials", -1, "number of iterations");
    parser.set_optional<std::string>("name_format",
                                     "name_format",
                                     "human",
                                     "either: json,human,txt");
    parser.set_optional<std::string>("seed", "seed", "random", get_seed_message());
    parser.run_and_exit_if_error();

    // Parse argv
    benchmark::Initialize(&argc, argv);
    const size_t bytes   = parser.get<size_t>("size");
    const int    trials = parser.get<int>("trials");
    bench_naming::set_format(parser.get<std::string>("name_format"));
    const std::string  seed_type = parser.get<std::string>("seed");
    const managed_seed seed(seed_type);

    // HIP
    hipStream_t stream = 0; // default

    // Benchmark info
    add_common_benchmark_info();
    benchmark::AddCustomContext("bytes", std::to_string(bytes));
    benchmark::AddCustomContext("seed", seed_type);

    using custom_128 = custom_type<int64_t, int64_t>;

    // Add benchmarks
    std::vector<benchmark::internal::Benchmark*> benchmarks = {CREATE_TYPED_BENCHMARK(int8_t),
                                                               CREATE_TYPED_BENCHMARK(int16_t),
                                                               CREATE_TYPED_BENCHMARK(int32_t),
                                                               CREATE_TYPED_BENCHMARK(int64_t),
                                                               CREATE_TYPED_BENCHMARK(custom_128)};

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
