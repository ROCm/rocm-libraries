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
#include "../../../bench_utils/bench_utils.hpp"

// rocThrust
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

// Google Benchmark
#include <benchmark/benchmark.h>

// STL
#include <cstddef>
#include <string>
#include <vector>

struct by_key
{
  template <typename KeyT, typename ValueT, typename Policy>
  float64_t run(thrust::device_vector<KeyT>& keys,
                thrust::device_vector<ValueT>& in_vals,
                thrust::device_vector<ValueT>& out_vals,
                Policy policy)
  {
    thrust::exclusive_scan_by_key(policy, keys.cbegin(), keys.cend(), in_vals.cbegin(), out_vals.begin());

    bench_utils::gpu_timer d_timer;

    d_timer.start(0);
    thrust::exclusive_scan_by_key(policy, keys.cbegin(), keys.cend(), in_vals.cbegin(), out_vals.begin());
    d_timer.stop(0);

    return d_timer.get_duration();
  }
};

template <class Benchmark, class KeyT, class ValueT>
void run_benchmark(benchmark::State& state, const std::size_t elements, const std::string seed_type)
{
  // Benchmark object
  Benchmark benchmark{};

  // GPU times
  std::vector<double> gpu_times;

  // Generate input
  thrust::device_vector<KeyT> keys =
    bench_utils::generate.uniform.key_segments(elements, seed_type, 0, 5200 /*magic numbers in thrust*/);
  thrust::device_vector<ValueT> in_vals(elements);

  // Output
  thrust::device_vector<ValueT> out_vals(elements);

  bench_utils::caching_allocator_t alloc{};
  thrust::detail::device_t policy{};

  for (auto _ : state)
  {
    float64_t duration = benchmark.template run<KeyT, ValueT>(keys, in_vals, out_vals, policy(alloc));
    state.SetIterationTime(duration);
    gpu_times.push_back(duration);
  }

  // BytesProcessed include read and written bytes, so when the BytesProcessed/s are reported
  // it will actually be the global memory bandwidth gotten.
  state.SetBytesProcessed(state.iterations() * (elements * (sizeof(KeyT) + 2 * sizeof(ValueT))));
  state.SetItemsProcessed(state.iterations() * elements);

  const double gpu_cv         = bench_utils::StatisticsCV(gpu_times);
  state.counters["gpu_noise"] = gpu_cv;
}

#define CREATE_BENCHMARK(KeyT, ValueT, Elements)                                                                     \
  benchmark::RegisterBenchmark(                                                                                      \
    bench_utils::bench_naming::format_name(                                                                          \
      "{algo:exclusive_scan,subalgo:" + name + ",key_type:" #KeyT + ",value_type:" #ValueT + ",elements:" #Elements) \
      .c_str(),                                                                                                      \
    run_benchmark<Benchmark, KeyT, ValueT>,                                                                          \
    Elements,                                                                                                        \
    seed_type)

#define BENCHMARK_VALUE_TYPE(key_type, value_type)                                                  \
  CREATE_BENCHMARK(key_type, value_type, 1 << 16), CREATE_BENCHMARK(key_type, value_type, 1 << 20), \
    CREATE_BENCHMARK(key_type, value_type, 1 << 24), CREATE_BENCHMARK(key_type, value_type, 1 << 28)

#if THRUST_BENCHMARKS_HAVE_INT128_SUPPORT
#  define BENCHMARK_KEY_TYPE(key_type)                                                  \
    BENCHMARK_VALUE_TYPE(key_type, int8_t), BENCHMARK_VALUE_TYPE(key_type, int16_t),    \
      BENCHMARK_VALUE_TYPE(key_type, int32_t), BENCHMARK_VALUE_TYPE(key_type, int64_t), \
      BENCHMARK_VALUE_TYPE(key_type, int128_t)
#else
#  define BENCHMARK_KEY_TYPE(key_type)                                               \
    BENCHMARK_VALUE_TYPE(key_type, int8_t), BENCHMARK_VALUE_TYPE(key_type, int16_t), \
      BENCHMARK_VALUE_TYPE(key_type, int32_t), BENCHMARK_VALUE_TYPE(key_type, int64_t)
#endif

template <class Benchmark>
void add_benchmarks(
  const std::string& name, std::vector<benchmark::internal::Benchmark*>& benchmarks, const std::string seed_type)
{
  std::vector<benchmark::internal::Benchmark*> bs = {
    BENCHMARK_KEY_TYPE(int8_t),
    BENCHMARK_KEY_TYPE(int16_t),
    BENCHMARK_KEY_TYPE(int32_t),
    BENCHMARK_KEY_TYPE(int64_t),
#if THRUST_BENCHMARKS_HAVE_INT128_SUPPORT
    BENCHMARK_KEY_TYPE(int128_t),
#endif
    BENCHMARK_KEY_TYPE(float32_t),
    BENCHMARK_KEY_TYPE(float64_t)
  };
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
  add_benchmarks<by_key>("by_key", benchmarks, seed_type);

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
