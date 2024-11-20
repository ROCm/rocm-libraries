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

// Benchmark utils
#include "../../bench_utils/bench_utils.hpp"

// rocThrust
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/reverse_iterator.h>

// Google Benchmark
#include <benchmark/benchmark.h>

// STL
#include <cstdlib>
#include <string>
#include <vector>

template <class T>
struct less_then_t
{
    T m_val;

    __host__ __device__ bool operator()(const T& val) const
    {
        return val < m_val;
    }
};

struct basic
{
    template <typename T, typename Policy>
    float64_t run(thrust::device_vector<T>& input,
                  thrust::device_vector<T>& output,
                  const std::size_t         elements,
                  less_then_t<T>            select_op,
                  Policy                    policy)
    {
        bench_utils::gpu_timer d_timer;

        d_timer.start(0);
        thrust::copy_if(policy,
                        input.cbegin(),
                        input.cend(),
                        output.begin(),
                        thrust::make_reverse_iterator(output.begin() + elements),
                        select_op);
        d_timer.stop(0);

        return d_timer.get_duration();
    }
};

template <class Benchmark, class T>
void run_benchmark(benchmark::State& state,
                   const std::size_t elements,
                   const std::string seed_type,
                   const int         entropy_reduction)
{
    // Benchmark object
    Benchmark benchmark {};

    // GPU times
    std::vector<double> gpu_times;

    // Generate input
    T val = bench_utils::value_from_entropy<T>(
        bench_utils::get_entropy_percentage(entropy_reduction));
    less_then_t<T> select_op {val};

    thrust::device_vector<T> input = bench_utils::generate(elements, seed_type);

    // Output
    thrust::device_vector<T> output(elements);

    bench_utils::caching_allocator_t alloc {};
    thrust::detail::device_t         policy {};

    for(auto _ : state)
    {
        float64_t duration
            = benchmark.template run<T>(input, output, elements, select_op, policy(alloc));
        state.SetIterationTime(duration);
        gpu_times.push_back(duration);
    }

    // BytesProcessed include read and written bytes, so when the BytesProcessed/s are reported
    // it will actually be the global memory bandwidth gotten.
    state.SetBytesProcessed(state.iterations() * 2 * elements * sizeof(T));
    state.SetItemsProcessed(state.iterations() * elements);

    const double gpu_cv         = bench_utils::StatisticsCV(gpu_times);
    state.counters["gpu_noise"] = gpu_cv;
}

#define CREATE_BENCHMARK(T, Elements, EntropyReduction)                                            \
    benchmark::RegisterBenchmark(                                                                  \
        bench_utils::bench_naming::format_name(                                                    \
            "{algo:partition,subalgo:" + name + ",input_type:" #T + ",elements:" #Elements         \
            + ",entropy:" + std::to_string(bench_utils::get_entropy_percentage(EntropyReduction))) \
            .c_str(),                                                                              \
        run_benchmark<Benchmark, T>,                                                               \
        Elements,                                                                                  \
        seed_type,                                                                                 \
        EntropyReduction)

#define BENCHMARK_TYPE_ENTROPY(type, entropy)                                           \
    CREATE_BENCHMARK(type, 1 << 16, entropy), CREATE_BENCHMARK(type, 1 << 20, entropy), \
        CREATE_BENCHMARK(type, 1 << 24, entropy), CREATE_BENCHMARK(type, 1 << 28, entropy)

template <class Benchmark>
void add_benchmarks(const std::string&                            name,
                    std::vector<benchmark::internal::Benchmark*>& benchmarks,
                    const std::string                             seed_type)
{
    constexpr int entropy_reductions[] = {0, 2, 6}; // 1.000, 0.544, 0.000;

    for(int entropy_reduction : entropy_reductions)
    {
        std::vector<benchmark::internal::Benchmark*> bs
            = {BENCHMARK_TYPE_ENTROPY(int8_t, entropy_reduction),
               BENCHMARK_TYPE_ENTROPY(int16_t, entropy_reduction),
               BENCHMARK_TYPE_ENTROPY(int32_t, entropy_reduction),
               BENCHMARK_TYPE_ENTROPY(int64_t, entropy_reduction),
               BENCHMARK_TYPE_ENTROPY(float, entropy_reduction),
               BENCHMARK_TYPE_ENTROPY(double, entropy_reduction)};
        benchmarks.insert(benchmarks.end(), bs.begin(), bs.end());
    }
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
