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
#include <thrust/partition.h>

// Google Benchmark
#include <benchmark/benchmark.h>

// STL
#include <cstddef>
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
                const std::size_t elements,
                less_then_t<T> select_op,
                Policy policy)
  {
    thrust::partition_copy(
      policy,
      input.cbegin(),
      input.cend(),
      output.begin(),
      thrust::make_reverse_iterator(output.begin() + elements),
      select_op);

    bench_utils::gpu_timer d_timer;

    d_timer.start(0);
    thrust::partition_copy(
      policy,
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
void run_benchmark(
  benchmark::State& state, const std::size_t elements, const std::string seed_type, const int entropy_reduction)
{
  using select_op_t = less_then_t<T>;

  // Benchmark object
  Benchmark benchmark{};

  // GPU times
  std::vector<double> gpu_times;

  // Generate input
  T val = bench_utils::value_from_entropy<T>(bench_utils::get_entropy_percentage(entropy_reduction));
  select_op_t select_op{val};

  thrust::device_vector<T> input = bench_utils::generate(elements, seed_type);

  // Output
  thrust::device_vector<T> output(elements);

  bench_utils::caching_allocator_t alloc{};
  thrust::detail::device_t policy{};

  for (auto _ : state)
  {
    float64_t duration = benchmark.template run<T>(input, output, elements, select_op, policy(alloc));
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

#define CREATE_BENCHMARK(T, Elements, EntropyReduction)                                      \
  benchmark::RegisterBenchmark(                                                              \
    bench_utils::bench_naming::format_name(                                                  \
      "{algo:partition,subalgo:" + name + ",input_type:" #T + ",elements:" #Elements         \
      + ",entropy:" + std::to_string(bench_utils::get_entropy_percentage(EntropyReduction))) \
      .c_str(),                                                                              \
    run_benchmark<Benchmark, T>,                                                             \
    Elements,                                                                                \
    seed_type,                                                                               \
    EntropyReduction)

#define BENCHMARK_TYPE_ENTROPY(type, entropy)                                         \
  CREATE_BENCHMARK(type, 1 << 16, entropy), CREATE_BENCHMARK(type, 1 << 20, entropy), \
    CREATE_BENCHMARK(type, 1 << 24, entropy), CREATE_BENCHMARK(type, 1 << 28, entropy)

template <class Benchmark>
void add_benchmarks(
  const std::string& name, std::vector<benchmark::internal::Benchmark*>& benchmarks, const std::string seed_type)
{
  constexpr int entropy_reductions[] = {0, 2, 4200}; // 1.000, 0.544, 0.000;

  for (int entropy_reduction : entropy_reductions)
  {
    std::vector<benchmark::internal::Benchmark*> bs = {
      BENCHMARK_TYPE_ENTROPY(int8_t, entropy_reduction),
      BENCHMARK_TYPE_ENTROPY(int16_t, entropy_reduction),
      BENCHMARK_TYPE_ENTROPY(int32_t, entropy_reduction),
      BENCHMARK_TYPE_ENTROPY(int64_t, entropy_reduction),
#if THRUST_BENCHMARKS_HAVE_INT128_SUPPORT
      BENCHMARK_TYPE_ENTROPY(int128_t, entropy_reduction),
#endif
      BENCHMARK_TYPE_ENTROPY(float, entropy_reduction),
      BENCHMARK_TYPE_ENTROPY(double, entropy_reduction)
    };
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
  add_benchmarks<basic>("basic", benchmarks, seed_type);

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
