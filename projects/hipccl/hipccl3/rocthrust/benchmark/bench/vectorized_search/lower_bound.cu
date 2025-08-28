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

// Benchmark utils
#include "../../bench_utils/bench_utils.hpp"

// rocThrust
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

// Google Benchmark
#include <benchmark/benchmark.h>

// STL
#include <cstddef>
#include <string>
#include <vector>

struct lower_bound
{
  template <typename T, typename Policy>
  float64_t
  run(thrust::device_vector<T>& data, thrust::device_vector<T>& result, const std::size_t elements, Policy policy)
  {
    thrust::lower_bound(
      policy, data.begin(), data.begin() + elements, data.begin() + elements, data.end(), result.begin());

    bench_utils::gpu_timer d_timer;

    d_timer.start(0);
    thrust::lower_bound(
      policy, data.begin(), data.begin() + elements, data.begin() + elements, data.end(), result.begin());
    d_timer.stop(0);

    return d_timer.get_duration();
  }
};

template <class Benchmark, class T>
void run_benchmark(
  benchmark::State& state, const std::size_t elements, const std::string seed_type, const std::size_t needles_ratio)
{
  // Benchmark object
  Benchmark benchmark{};

  // GPU times
  std::vector<double> gpu_times;

  // Generate input
  const auto needles = needles_ratio * static_cast<std::size_t>(static_cast<double>(elements) / 100.0);

  thrust::device_vector<T> data = bench_utils::generate(elements + needles, seed_type);
  thrust::device_vector<T> result(needles);
  thrust::sort(data.begin(), data.begin() + elements);

  bench_utils::caching_allocator_t alloc{};
  thrust::detail::device_t policy{};

  for (auto _ : state)
  {
    float64_t duration = benchmark.template run<T>(data, result, elements, policy(alloc));
    state.SetIterationTime(duration);
    gpu_times.push_back(duration);
  }

  state.SetBytesProcessed(0);
  state.SetItemsProcessed(state.iterations() * needles);

  const double gpu_cv         = bench_utils::StatisticsCV(gpu_times);
  state.counters["gpu_noise"] = gpu_cv;
}

#define CREATE_BENCHMARK(T, Elements, NeedlesRatio)                                                      \
  benchmark::RegisterBenchmark(                                                                          \
    bench_utils::bench_naming::format_name("{algo:vectorized_search,subalgo:" + name + ",input_type:" #T \
                                           + ",elements:" #Elements + ",needles_ratio:" #NeedlesRatio)   \
      .c_str(),                                                                                          \
    run_benchmark<Benchmark, T>,                                                                         \
    Elements,                                                                                            \
    seed_type,                                                                                           \
    NeedlesRatio)

#define BENCHMARK_ELEMENTS(type, elements) \
  CREATE_BENCHMARK(type, elements, 1), CREATE_BENCHMARK(type, elements, 25), CREATE_BENCHMARK(type, elements, 50)

#define BENCHMARK_TYPE(type)                                                                               \
  BENCHMARK_ELEMENTS(type, 1 << 16), BENCHMARK_ELEMENTS(type, 1 << 20), BENCHMARK_ELEMENTS(type, 1 << 24), \
    BENCHMARK_ELEMENTS(type, 1 << 28)

template <class Benchmark>
void add_benchmarks(
  const std::string& name, std::vector<benchmark::internal::Benchmark*>& benchmarks, const std::string seed_type)
{
  std::vector<benchmark::internal::Benchmark*> bs = {
    BENCHMARK_TYPE(int8_t), BENCHMARK_TYPE(int16_t), BENCHMARK_TYPE(int32_t), BENCHMARK_TYPE(int64_t)};
  benchmarks.insert(benchmarks.end(), bs.begin(), bs.end());
}

int main(int argc, char* argv[])
{
  cli::Parser parser(argc, argv);
  parser.set_optional<std::string>("name_format", "name_format", "human", "either: json,human,txt");
  parser.set_optional<std::string>("seed", "seed", "random", bench_utils::get_seed_message());
  parser.run_and_exit_if_error();

  // Parse argv
  benchmark::Initialize(&argc, argv);
  bench_utils::bench_naming::set_format(parser.get<std::string>("name_format")); /* either: json,human,txt */
  const std::string seed_type = parser.get<std::string>("seed");

  // Benchmark info
  bench_utils::add_common_benchmark_info();
  benchmark::AddCustomContext("seed", seed_type);

  // Add benchmark
  std::vector<benchmark::internal::Benchmark*> benchmarks;
  add_benchmarks<lower_bound>("lower_bound", benchmarks, seed_type);

  // Use manual timing
  for (auto& b : benchmarks)
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
