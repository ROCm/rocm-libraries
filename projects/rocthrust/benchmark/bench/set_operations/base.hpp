/******************************************************************************
 * Copyright (c) 2011-2023, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2024-2025, Advanced Micro Devices, Inc.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#pragma once

// Benchmark utils
#include "../../bench_utils/bench_utils.hpp"

// rocThrust
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/set_operations.h>
#include <thrust/sort.h>

// Google Benchmark
#include <benchmark/benchmark.h>

// STL
#include <cstddef>
#include <string>
#include <vector>

struct basic
{
  template <typename T, typename OpT, typename Policy>
  float64_t run(thrust::device_vector<T>& input,
                thrust::device_vector<T>& output,
                const std::size_t elements_in_A,
                const OpT op,
                Policy policy)
  {
    bench_utils::gpu_timer d_timer;

    d_timer.start(0);
    op(policy,
       input.cbegin(),
       input.cbegin() + elements_in_A,
       input.cbegin() + elements_in_A,
       input.cend(),
       output.begin());
    d_timer.stop(0);

    return d_timer.get_duration();
  }
};

template <class T, class OpT>
void run_benchmark(benchmark::State& state,
                   const std::size_t elements,
                   const std::string seed_type,
                   const int entropy_reduction,
                   const std::size_t input_size_ratio)
{
  // Benchmark object
  basic benchmark{};

  // GPU times
  std::vector<double> gpu_times;

  // Generate input
  const auto entropy       = bench_utils::get_entropy_percentage(entropy_reduction) / 100.0f;
  const auto elements_in_A = static_cast<std::size_t>(static_cast<double>(input_size_ratio * elements) / 100.0f);

  thrust::device_vector<T> input = bench_utils::generate(elements, seed_type, entropy);
  thrust::device_vector<T> output(elements);

  thrust::sort(input.begin(), input.begin() + elements_in_A);
  thrust::sort(input.begin() + elements_in_A, input.end());

  OpT op{};

  bench_utils::caching_allocator_t alloc{};
  thrust::detail::device_t policy{};
  // not a warm-up run, we need to run once to determine the size of the output
  const auto result_ends =
    op(policy(alloc),
       input.cbegin(),
       input.cbegin() + elements_in_A,
       input.cbegin() + elements_in_A,
       input.cend(),
       output.begin());
  const std::size_t elements_in_AB = thrust::distance(output.begin(), result_ends);

  for (auto _ : state)
  {
    float64_t duration = benchmark.template run<T>(input, output, elements_in_A, op, policy(alloc));
    state.SetIterationTime(duration);
    gpu_times.push_back(duration);
  }

  // BytesProcessed include read and written bytes, so when the BytesProcessed/s are reported
  // it will actually be the global memory bandwidth gotten.
  state.SetBytesProcessed(state.iterations() * (elements + elements_in_AB) * sizeof(T));
  state.SetItemsProcessed(state.iterations() * elements);

  const double gpu_cv         = bench_utils::StatisticsCV(gpu_times);
  state.counters["gpu_noise"] = gpu_cv;
}

#define CREATE_BENCHMARK(T, Elements, EntropyReduction, InputSizeRatio)                                               \
  benchmark::RegisterBenchmark(                                                                                       \
    bench_utils::bench_naming::format_name(                                                                           \
      "{algo:" + algo_name + ",subalgo:basic" + ",input_type:" #T + ",elements:" #Elements + ",entropy:"              \
      + std::to_string(bench_utils::get_entropy_percentage(EntropyReduction)) + ",input_size_ratio:" #InputSizeRatio) \
      .c_str(),                                                                                                       \
    run_benchmark<T, OpT>,                                                                                            \
    Elements,                                                                                                         \
    seed_type,                                                                                                        \
    EntropyReduction,                                                                                                 \
    InputSizeRatio)

#define BENCHMARK_ELEMENTS(type, elements, entropy)                                             \
  CREATE_BENCHMARK(type, elements, entropy, 25), CREATE_BENCHMARK(type, elements, entropy, 50), \
    CREATE_BENCHMARK(type, elements, entropy, 75)

#define BENCHMARK_TYPE_ENTROPY(type, entropy)                                             \
  BENCHMARK_ELEMENTS(type, 1 << 16, entropy), BENCHMARK_ELEMENTS(type, 1 << 20, entropy), \
    BENCHMARK_ELEMENTS(type, 1 << 24, entropy), BENCHMARK_ELEMENTS(type, 1 << 28, entropy)

template <class OpT>
void add_benchmarks(
  const std::string& algo_name, std::vector<benchmark::internal::Benchmark*>& benchmarks, const std::string seed_type)
{
  constexpr int entropy_reductions[] = {0, 4}; // 1.000, 0.201;

  for (int entropy_reduction : entropy_reductions)
  {
    std::vector<benchmark::internal::Benchmark*> bs = {
      BENCHMARK_TYPE_ENTROPY(int8_t, entropy_reduction),
      BENCHMARK_TYPE_ENTROPY(int16_t, entropy_reduction),
      BENCHMARK_TYPE_ENTROPY(int32_t, entropy_reduction),
      BENCHMARK_TYPE_ENTROPY(int64_t, entropy_reduction)};

    benchmarks.insert(benchmarks.end(), bs.begin(), bs.end());
  }
}
