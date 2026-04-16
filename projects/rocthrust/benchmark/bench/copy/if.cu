/******************************************************************************
 * Copyright (c) 2011-2023, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2024-2026, Advanced Micro Devices, Inc.  All rights reserved.
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
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

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

struct _if
{
  template <typename T, typename Policy>
  float64_t
  run(thrust::device_vector<T>& input, thrust::device_vector<T>& output, less_then_t<T> select_op, Policy policy)
  {
    thrust::copy_if(policy, input.cbegin(), input.cend(), output.begin(), select_op);

    bench_utils::gpu_timer d_timer;

    d_timer.start(0);
    thrust::copy_if(policy, input.cbegin(), input.cend(), output.begin(), select_op);
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

  T val = bench_utils::value_from_entropy<T>(bench_utils::get_entropy_percentage(entropy_reduction));
  select_op_t select_op{val};

  // Generate input and output
  thrust::device_vector<T> input;
  try
  {
    input = bench_utils::generate(elements, seed_type);
  }
  catch (const ::thrust::system::detail::bad_alloc& e)
  {
    (void) hipGetLastError();
    state.SkipWithError(("thrust::system::detail::bad_alloc: " + std::string(e.what())).c_str());
    return;
  }
  const auto selected_elements = thrust::count_if(input.cbegin(), input.cend(), select_op);
  thrust::device_vector<T> output;
  try
  {
    output = thrust::device_vector<T>(selected_elements);
  }
  catch (const ::thrust::system::detail::bad_alloc& e)
  {
    (void) hipGetLastError();
    state.SkipWithError(("thrust::system::detail::bad_alloc: " + std::string(e.what())).c_str());
    return;
  }

  bench_utils::caching_allocator_t alloc{};
  thrust::detail::device_t policy{};

  for (auto _ : state)
  {
    float64_t duration = benchmark.template run<T>(input, output, select_op, policy(alloc));
    state.SetIterationTime(duration);
    gpu_times.push_back(duration);
  }

  // BytesProcessed include read and written bytes, so when the BytesProcessed/s are reported
  // it will actually be the global memory bandwidth gotten.
  state.SetBytesProcessed(state.iterations() * (elements + selected_elements) * sizeof(T));
  state.SetItemsProcessed(state.iterations() * elements);

  const double gpu_cv         = bench_utils::StatisticsCV(gpu_times);
  state.counters["gpu_noise"] = gpu_cv;
}

#define CREATE_BENCHMARK(T, Elements, EntropyReduction)                                                    \
  benchmark::RegisterBenchmark(                                                                            \
    bench_utils::bench_naming::format_name(                                                                \
      "{algo:copy,subalgo:" + name + ",input_type:" #T + ",elements:" + bench_utils::format_pow2(Elements) \
      + ",entropy:" + std::to_string(bench_utils::get_entropy_percentage(EntropyReduction)))               \
      .c_str(),                                                                                            \
    run_benchmark<Benchmark, T>,                                                                           \
    Elements,                                                                                              \
    seed_type,                                                                                             \
    EntropyReduction)

#define BENCHMARK_TYPE_ENTROPY(type, entropies)                              \
  for (size_t size : bench_utils::sizes)                                     \
  {                                                                          \
    for (int entropy : entropies)                                            \
    {                                                                        \
      if (sizeof(type) * size <= bench_utils::system.devProp.totalGlobalMem) \
        bs.push_back(CREATE_BENCHMARK(type, size, entropy));                 \
    }                                                                        \
  }

template <class Benchmark>
void add_benchmarks(const std::string& name, std::vector<benchmark::Benchmark*>& benchmarks, const std::string seed_type)
{
  constexpr int entropy_reductions[] = {0, 2, 4200}; // 1.000, 0.544, 0.000;
  std::vector<benchmark::Benchmark*> bs;

  BENCHMARK_TYPE_ENTROPY(int8_t, entropy_reductions)
  BENCHMARK_TYPE_ENTROPY(int16_t, entropy_reductions)
  BENCHMARK_TYPE_ENTROPY(int32_t, entropy_reductions)
  BENCHMARK_TYPE_ENTROPY(int64_t, entropy_reductions)
  BENCHMARK_TYPE_ENTROPY(float, entropy_reductions)
  BENCHMARK_TYPE_ENTROPY(double, entropy_reductions)
  BENCHMARK_TYPE_ENTROPY(bench_utils::large_data, entropy_reductions)

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
  std::vector<benchmark::Benchmark*> benchmarks;
  add_benchmarks<_if>("if", benchmarks, seed_type);

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
