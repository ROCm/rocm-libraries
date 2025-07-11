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
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

// Google Benchmark
#include <benchmark/benchmark.h>

// STL
#include <cstddef>
#include <string>
#include <vector>

struct pairs_custom
{
  template <typename KeyT, typename ValueT, typename Policy>
  float64_t run(thrust::device_vector<KeyT>& in_keys, thrust::device_vector<ValueT>& in_vals, Policy policy)
  {
    thrust::device_vector<KeyT> keys(in_keys.size());
    thrust::device_vector<ValueT> vals(in_vals.size());
    thrust::sort_by_key(policy, keys.begin(), keys.end(), vals.begin(), bench_utils::less_t{});

    keys = in_keys;
    vals = in_vals;

    bench_utils::gpu_timer d_timer;

    d_timer.start(0);
    thrust::sort_by_key(policy, keys.begin(), keys.end(), vals.begin(), bench_utils::less_t{});
    d_timer.stop(0);

    return d_timer.get_duration();
  }
};

template <class Benchmark, class KeyT, class ValueT>
void run_benchmark(
  benchmark::State& state, const std::size_t elements, const std::string seed_type, const int entropy_reduction)
{
  // Benchmark object
  Benchmark benchmark{};

  // GPU times
  std::vector<double> gpu_times;

  // Generate input
  const auto entropy                 = bench_utils::get_entropy_percentage(entropy_reduction) / 100.0f;
  thrust::device_vector<KeyT> in_keys   = bench_utils::generate(elements, seed_type, entropy);
  thrust::device_vector<ValueT> in_vals = bench_utils::generate(elements, seed_type);

  bench_utils::caching_allocator_t alloc{};
  thrust::detail::device_t policy{};

  for (auto _ : state)
  {
    float64_t duration = benchmark.template run<KeyT, ValueT>(in_keys, in_vals, policy(alloc));
    state.SetIterationTime(duration);
    gpu_times.push_back(duration);
  }

  // BytesProcessed include read and written bytes, so when the BytesProcessed/s are reported
  // it will actually be the global memory bandwidth gotten.
  state.SetBytesProcessed(state.iterations() * 2 * elements * (sizeof(KeyT) + sizeof(ValueT)));
  state.SetItemsProcessed(state.iterations() * elements);

  const double gpu_cv         = bench_utils::StatisticsCV(gpu_times);
  state.counters["gpu_noise"] = gpu_cv;
}

#define CREATE_BENCHMARK(KeyT, ValueT, Elements, EntropyReduction)                                        \
  benchmark::RegisterBenchmark(                                                                           \
    bench_utils::bench_naming::format_name(                                                               \
      "{algo:sort,subalgo:" + name + ",key_type:" #KeyT + ",value_type:" #ValueT + ",elements:" #Elements \
      + ",entropy:" + std::to_string(bench_utils::get_entropy_percentage(EntropyReduction)))              \
      .c_str(),                                                                                           \
    run_benchmark<Benchmark, KeyT, ValueT>,                                                               \
    Elements,                                                                                             \
    seed_type,                                                                                            \
    EntropyReduction)

#define BENCHMARK_VALUE_TYPE(key_type, value_type, entropy)                                                           \
  CREATE_BENCHMARK(key_type, value_type, 1 << 16, entropy), CREATE_BENCHMARK(key_type, value_type, 1 << 20, entropy), \
    CREATE_BENCHMARK(key_type, value_type, 1 << 24, entropy), CREATE_BENCHMARK(key_type, value_type, 1 << 28, entropy)

#define BENCHMARK_KEY_TYPE_ENTROPY(key_type, entropy)                                                \
  BENCHMARK_VALUE_TYPE(key_type, int8_t, entropy), BENCHMARK_VALUE_TYPE(key_type, int16_t, entropy), \
    BENCHMARK_VALUE_TYPE(key_type, int32_t, entropy), BENCHMARK_VALUE_TYPE(key_type, int64_t, entropy)

template <class Benchmark>
void add_benchmarks(
  const std::string& name, std::vector<benchmark::internal::Benchmark*>& benchmarks, const std::string seed_type)
{
  constexpr int entropy_reductions[] = {0, 2, 6}; // 1.000, 0.544, 0.000;

  for (int entropy_reduction : entropy_reductions)
  {
    std::vector<benchmark::internal::Benchmark*> bs = {
      BENCHMARK_KEY_TYPE_ENTROPY(int8_t, entropy_reduction),
      BENCHMARK_KEY_TYPE_ENTROPY(int16_t, entropy_reduction),
      BENCHMARK_KEY_TYPE_ENTROPY(int32_t, entropy_reduction),
      BENCHMARK_KEY_TYPE_ENTROPY(int64_t, entropy_reduction)};
    benchmarks.insert(benchmarks.end(), bs.begin(), bs.end());
  }
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
  add_benchmarks<pairs_custom>("pairs_custom", benchmarks, seed_type);

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
