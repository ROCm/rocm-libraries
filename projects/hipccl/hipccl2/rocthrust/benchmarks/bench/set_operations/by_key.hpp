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

#ifndef ROCTHRUST_BASE_HPP_
#define ROCTHRUST_BASE_HPP_

// Benchmark utils
#include "../../bench_utils/bench_utils.hpp"

// rocThrust
#include <thrust/device_vector.h>
#include <thrust/distance.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

// Google Benchmark
#include <benchmark/benchmark.h>

// STL
#include <cstdlib>
#include <string>
#include <vector>

struct by_key
{
    template <typename KeyT,
              typename ValueT,
              typename OpT,
              typename Policy = thrust::detail::device_t>
    float64_t run(thrust::device_vector<KeyT>&   input_keys,
                  thrust::device_vector<ValueT>& input_vals,
                  thrust::device_vector<KeyT>&   output_keys,
                  thrust::device_vector<ValueT>& output_vals,
                  const std::size_t              elements_in_A,
                  const OpT                      op)
    {
        bench_utils::gpu_timer d_timer;

        d_timer.start(0);
        op(Policy {},
           input_keys.cbegin(),
           input_keys.cbegin() + elements_in_A,
           input_keys.cbegin() + elements_in_A,
           input_keys.cend(),
           input_vals.cbegin(),
           input_vals.cbegin() + elements_in_A,
           output_keys.begin(),
           output_vals.begin());
        d_timer.stop(0);

        return d_timer.get_duration();
    }
};

template <class KeyT, class ValueT, class OpT>
void run_benchmark(benchmark::State& state,
                   const std::size_t elements,
                   const std::string seed_type,
                   const int         entropy_reduction,
                   const std::size_t input_size_ratio)
{
    // Benchmark object
    by_key benchmark {};

    // GPU times
    std::vector<double> gpu_times;

    // Generate input
    const auto entropy = bench_utils::get_entropy_percentage(entropy_reduction) / 100.0f;
    const auto elements_in_A
        = static_cast<std::size_t>(static_cast<double>(input_size_ratio * elements) / 100.0f);

    thrust::device_vector<KeyT> input_keys = bench_utils::generate(elements, seed_type, entropy);
    thrust::device_vector<KeyT> output_keys(elements);

    thrust::device_vector<ValueT> input_vals(elements);
    thrust::device_vector<ValueT> output_vals(elements);

    thrust::sort(input_keys.begin(), input_keys.begin() + elements_in_A);
    thrust::sort(input_keys.begin() + elements_in_A, input_keys.end());

    OpT  op {};
    auto result_ends = op(thrust::detail::device_t {},
                          input_keys.cbegin(),
                          input_keys.cbegin() + elements_in_A,
                          input_keys.cbegin() + elements_in_A,
                          input_keys.cend(),
                          input_vals.cbegin(),
                          input_vals.cbegin() + elements_in_A,
                          output_keys.begin(),
                          output_vals.begin());

    const std::size_t elements_in_AB = thrust::distance(output_keys.begin(), result_ends.first);

    for(auto _ : state)
    {
        float64_t duration = benchmark.template run<KeyT, ValueT, OpT>(
            input_keys, input_vals, output_keys, output_vals, elements_in_A, op);
        state.SetIterationTime(duration);
        gpu_times.push_back(duration);
    }

    // BytesProcessed include read and written bytes, so when the BytesProcessed/s are reported
    // it will actually be the global memory bandwidth gotten.
    const std::size_t global_memory_key_bytes   = (elements + elements_in_AB) * sizeof(KeyT);
    const std::size_t global_memory_value_reads = OpT::read_all_values ? elements : elements_in_A;
    const std::size_t global_memory_value_bytes
        = (global_memory_value_reads + elements_in_AB) * sizeof(ValueT);
    state.SetBytesProcessed(state.iterations()
                            * (global_memory_key_bytes + global_memory_value_bytes));
    state.SetItemsProcessed(state.iterations() * elements);

    const double gpu_cv         = bench_utils::StatisticsCV(gpu_times);
    state.counters["gpu_noise"] = gpu_cv;
}

#define CREATE_BENCHMARK(KeyT, ValueT, Elements, EntropyReduction, InputSizeRatio)                 \
    benchmark::RegisterBenchmark(                                                                  \
        bench_utils::bench_naming::format_name(                                                    \
            "{algo:" + algo_name + ",subalgo:by_key" + ",key_type:" #KeyT + ",value_type:" #ValueT \
            + ",elements:" #Elements                                                               \
            + ",entropy:" + std::to_string(bench_utils::get_entropy_percentage(EntropyReduction))  \
            + ",input_size_ratio:" #InputSizeRatio)                                                \
            .c_str(),                                                                              \
        run_benchmark<KeyT, ValueT, OpT>,                                                          \
        Elements,                                                                                  \
        seed_type,                                                                                 \
        EntropyReduction,                                                                          \
        InputSizeRatio)

#define BENCHMARK_ELEMENTS(key_type, value_type, elements, entropy)    \
    CREATE_BENCHMARK(key_type, value_type, elements, entropy, 25),     \
        CREATE_BENCHMARK(key_type, value_type, elements, entropy, 50), \
        CREATE_BENCHMARK(key_type, value_type, elements, entropy, 75)

#define BENCHMARK_VALUE_TYPE(key_type, value_type, entropy)         \
    BENCHMARK_ELEMENTS(key_type, value_type, 1 << 16, entropy),     \
        BENCHMARK_ELEMENTS(key_type, value_type, 1 << 20, entropy), \
        BENCHMARK_ELEMENTS(key_type, value_type, 1 << 24, entropy), \
        BENCHMARK_ELEMENTS(key_type, value_type, 1 << 28, entropy)

#define BENCHMARK_KEY_TYPE_ENTROPY(key_type, entropy) \
    BENCHMARK_VALUE_TYPE(key_type, int8_t, entropy),  \
        BENCHMARK_VALUE_TYPE(key_type, int64_t, entropy)

template <class OpT>
void add_benchmarks(const std::string&                            algo_name,
                    std::vector<benchmark::internal::Benchmark*>& benchmarks,
                    const std::string                             seed_type)
{
    constexpr int entropy_reductions[] = {0, 4}; // 1.000, 0.201;

    for(int entropy_reduction : entropy_reductions)
    {
        std::vector<benchmark::internal::Benchmark*> bs
            = {BENCHMARK_KEY_TYPE_ENTROPY(int8_t, entropy_reduction),
               BENCHMARK_KEY_TYPE_ENTROPY(int16_t, entropy_reduction),
               BENCHMARK_KEY_TYPE_ENTROPY(int32_t, entropy_reduction),
               BENCHMARK_KEY_TYPE_ENTROPY(int64_t, entropy_reduction)};

        benchmarks.insert(benchmarks.end(), bs.begin(), bs.end());
    }
}

#endif // ROCTHRUST_BASE_HPP_
