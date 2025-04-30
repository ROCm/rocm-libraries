// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

// Benchmark utils
#include "../../bench_utils/bench_utils.hpp"

// rocThrust
#include <thrust/equal.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

// Google Benchmark
#include <benchmark/benchmark.h>

// STL
#include <cstdlib>
#include <string>
#include <vector>

struct basic
{
    template <typename T, typename Policy>
    float64_t run(thrust::device_vector<T>& a, thrust::device_vector<T>& b, Policy policy)
    {
        bench_utils::gpu_timer d_timer;

        d_timer.start(0);
        thrust::equal(policy, a.begin(), a.end(), b.begin());
        d_timer.stop(0);

        return d_timer.get_duration();
    }
};

template <class Benchmark, class T>
void run_benchmark(benchmark::State& state,
                   const std::size_t elements,
                   const std::string , /*seed_type*/
                   const float64_t common_prefix_ratio)
{
    // Benchmark object
    Benchmark benchmark {};

    // GPU times
    std::vector<double> gpu_times;

    thrust::device_vector<T> a(elements, T{1});
    thrust::device_vector<T> b(elements, T{1});

    const auto same_elements = std::min(static_cast<std::size_t>(elements * common_prefix_ratio), elements);

    bench_utils::caching_allocator_t alloc;
    thrust::detail::device_t         policy {};

    thrust::fill(policy(alloc), b.begin() + same_elements, b.end(), T{2});

    for(auto _ : state)
    {
        float64_t duration = benchmark.template run<T>(a, b, policy(alloc));
        state.SetIterationTime(duration);
        gpu_times.push_back(duration);
    }

    // BytesProcessed include read and written bytes, so when the BytesProcessed/s are reported
    // it will actually be the global memory bandwidth gotten.
    // using `same_elements` instead of `elements` corresponds to the 
    // actual elements read in an early exit
    state.SetBytesProcessed(state.iterations() * 2 *
                            std::max(same_elements, std::size_t(1)) * sizeof(T));
    state.SetItemsProcessed(state.iterations() *
                            std::max(same_elements, std::size_t(1)));

    const double gpu_cv         = bench_utils::StatisticsCV(gpu_times);
    state.counters["gpu_noise"] = gpu_cv;
}

#define CREATE_BENCHMARK(T, Elements, CommonPrefixRatio)                                         \
    benchmark::RegisterBenchmark(                                                                \
        bench_utils::bench_naming::format_name("{algo:equal,subalgo:" + name + ",input_type:" #T \
                                               + ",elements:" #Elements +                        \
                                               ", common_prefix_ratio:" #CommonPrefixRatio)      \
            .c_str(),                                                                            \
        run_benchmark<Benchmark, T>,                                                             \
        Elements,                                                                                \
        seed_type,                                                                               \
        CommonPrefixRatio)

#define BENCHMARK_ELEMENTS(type, elements)     \
        CREATE_BENCHMARK(type, elements, 1.0), \
        CREATE_BENCHMARK(type, elements, 0.5), \
        CREATE_BENCHMARK(type, elements, 0.0)

#define BENCHMARK_TYPE(type)                                                  \
        BENCHMARK_ELEMENTS(type, 1 << 16), BENCHMARK_ELEMENTS(type, 1 << 20), \
        BENCHMARK_ELEMENTS(type, 1 << 24), BENCHMARK_ELEMENTS(type, 1 << 28)

template <class Benchmark>
void add_benchmarks(const std::string&                            name,
                    std::vector<benchmark::internal::Benchmark*>& benchmarks,
                    const std::string                             seed_type)
{
    std::vector<benchmark::internal::Benchmark*> bs = {BENCHMARK_TYPE(int8_t),
                                                       BENCHMARK_TYPE(int16_t),
                                                       BENCHMARK_TYPE(int32_t),
                                                       BENCHMARK_TYPE(int64_t)};

    benchmarks.insert(benchmarks.end(), bs.begin(), bs.end());
}

int main(int argc, char* argv[])
{
    cli::Parser parser(argc, argv);
    parser.set_optional<std::string>(
        "name_format", "name_format", "human", "either: json,human,txt");
    parser.set_optional<std::string>("seed", "seed", "random", bench_utils::get_seed_message());
    parser.run_and_exit_if_error();

    // Parse argv
    benchmark::Initialize(&argc, argv);
    bench_utils::bench_naming::set_format(
        parser.get<std::string>("name_format")); /* either: json,human,txt */
    const std::string seed_type = parser.get<std::string>("seed");

    // Benchmark info
    bench_utils::add_common_benchmark_info();
    benchmark::AddCustomContext("seed", seed_type);

    // Add benchmark
    std::vector<benchmark::internal::Benchmark*> benchmarks;
    add_benchmarks<basic>("basic", benchmarks, seed_type);

    // Use manual timing
    for(auto& b : benchmarks)
    {
        b->UseManualTime();
        b->Unit(benchmark::kMicrosecond);
        b->MinTime(0.4); // in seconds
    }

    // Run benchmarks
    benchmark::RunSpecifiedBenchmarks(bench_utils::ChooseCustomReporter());

    // Finish
    benchmark::Shutdown();
    return 0;
}
