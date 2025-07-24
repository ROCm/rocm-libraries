// MIT License
//
// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

// Google Benchmark
#include <benchmark/benchmark.h>

// STL
#include <cstdlib>
#include <string>
#include <vector>

struct pairs_custom
{
  template <typename KeyT, typename ValueT, typename Policy>
  float64_t run(thrust::device_vector<KeyT>& keys, thrust::device_vector<ValueT>& vals, Policy policy)
  {
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
  thrust::device_vector<KeyT> keys   = bench_utils::generate(elements, seed_type, entropy);
  thrust::device_vector<ValueT> vals = bench_utils::generate(elements, seed_type);

  bench_utils::caching_allocator_t alloc{};
  thrust::detail::device_t policy{};

  for (auto _ : state)
  {
    float64_t duration = benchmark.template run<KeyT, ValueT>(keys, vals, policy(alloc));
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
