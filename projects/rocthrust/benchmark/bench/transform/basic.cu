/******************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/detail/functional/address_stability.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/zip_function.h>

// Google Benchmark
#include <benchmark/benchmark.h>

// STL
#include <cstddef>
#include <string>
#include <vector>
#if !_THRUST_HAS_DEVICE_SYSTEM_STD
#  include <utility>
#endif

template <class InT, class OutT>
struct fib_t
{
  __device__ OutT operator()(InT n)
  {
    OutT t1 = 0;
    OutT t2 = 1;

    if (n < 1)
    {
      return t1;
    }
    else if (n == 1)
    {
      return t1;
    }
    else if (n == 2)
    {
      return t2;
    }
    for (InT i = 3; i <= n; ++i)
    {
      const auto next = t1 + t2;
      t1              = t2;
      t2              = next;
    }

    return t2;
  }
};

template <typename... Args>
float64_t bench_transform(Args&&... args)
{
  bench_utils::caching_allocator_t alloc{}; // transform shouldn't allocate, but let's be consistent
  thrust::detail::device_t policy{};
  thrust::transform(policy(alloc), _THRUST_STD::forward<Args>(args)...); // warmup (queries and caches occupancy)

  bench_utils::gpu_timer d_timer;
  d_timer.start(0);
  thrust::transform(policy(alloc), _THRUST_STD::forward<Args>(args)...);
  d_timer.stop(0);

  return d_timer.get_duration();
}

struct basic
{
  template <typename T>
  float64_t run(thrust::device_vector<T>& input, thrust::device_vector<T>& output)
  {
    fib_t<T, uint32_t> op{};
    return bench_transform(input.cbegin(), input.cend(), output.begin(), op);
  }
};

template <class Benchmark, class T>
void run_benchmark(benchmark::State& state, const std::size_t elements, const std::string seed_type)
{
  // Benchmark object
  Benchmark benchmark{};

  // GPU times
  std::vector<double> gpu_times;

  // Generate input
  thrust::device_vector<T> input = bench_utils::generate(
    elements,
    seed_type,
    0 /*entropy 1.000*/,
    T{0} /*magic number used in Thrust*/,
    T{42} /*magic number used in Thrust*/);
  thrust::device_vector<T> output(elements);

  for (auto _ : state)
  {
    float64_t duration = benchmark.template run<T>(input, output);
    state.SetIterationTime(duration);
    gpu_times.push_back(duration);
  }

  // BytesProcessed include read and written bytes, so when the BytesProcessed/s are reported
  // it will actually be the global memory bandwidth gotten.
  state.SetBytesProcessed(state.iterations() * elements * (sizeof(T) + sizeof(uint32_t)));
  state.SetItemsProcessed(state.iterations() * elements);

  const double gpu_cv         = bench_utils::StatisticsCV(gpu_times);
  state.counters["gpu_noise"] = gpu_cv;
}

#define CREATE_BENCHMARK(T, Elements)                                                 \
  benchmark::RegisterBenchmark(                                                       \
    bench_utils::bench_naming::format_name(                                           \
      "{algo:transform,subalgo:" + name + ",input_type:" #T + ",elements:" #Elements) \
      .c_str(),                                                                       \
    run_benchmark<Benchmark, T>,                                                      \
    Elements,                                                                         \
    seed_type)

// clang-format off
#define BENCHMARK_TYPE(type)       \
  CREATE_BENCHMARK(type, 1 << 16), \
  CREATE_BENCHMARK(type, 1 << 20), \
  CREATE_BENCHMARK(type, 1 << 24), \
  CREATE_BENCHMARK(type, 1 << 28)
// clang-format on

template <class Benchmark>
void add_benchmarks(
  const std::string& name, std::vector<benchmark::internal::Benchmark*>& benchmarks, const std::string seed_type)
{
  std::vector<benchmark::internal::Benchmark*> bs = {BENCHMARK_TYPE(uint32_t), BENCHMARK_TYPE(uint64_t)};

  benchmarks.insert(benchmarks.end(), bs.begin(), bs.end());
}

namespace babelstream
{
// The benchmarks in this namespace are inspired by the BabelStream thrust version:
// https://github.com/UoB-HPC/BabelStream/blob/main/src/thrust/ThrustStream.cu

// Modified from BabelStream to also work for integers
constexpr auto startA      = 1;
constexpr auto startB      = 2;
constexpr auto startC      = 3;
constexpr auto startScalar = 4;

struct mul
{
  static constexpr size_t reads_per_item  = 1;
  static constexpr size_t writes_per_item = 1;

  template <typename T>
  static float64_t run(thrust::device_vector<T>, thrust::device_vector<T> b, thrust::device_vector<T> c)
  {
    const T scalar = startScalar;
    return bench_transform(
      c.begin(), c.end(), b.begin(), ::thrust::detail::proclaim_copyable_arguments([=] THRUST_DEVICE(const T& ci) {
        return ci * scalar;
      }));
  }
};
struct add
{
  static constexpr size_t reads_per_item  = 2;
  static constexpr size_t writes_per_item = 1;

  template <typename T>
  static float64_t run(thrust::device_vector<T> a, thrust::device_vector<T> b, thrust::device_vector<T> c)
  {
    return bench_transform(
      a.begin(),
      a.end(),
      b.begin(),
      c.begin(),
      ::thrust::detail::proclaim_copyable_arguments([] THRUST_DEVICE(const T& ai, const T& bi) -> T {
        return ai + bi;
      }));
  }
};

struct triad
{
  static constexpr size_t reads_per_item  = 2;
  static constexpr size_t writes_per_item = 1;

  template <typename T>
  static float64_t run(thrust::device_vector<T> a, thrust::device_vector<T> b, thrust::device_vector<T> c)
  {
    const T scalar = startScalar;
    return bench_transform(
      b.begin(),
      b.end(),
      c.begin(),
      a.begin(),
      ::thrust::detail::proclaim_copyable_arguments([=] THRUST_DEVICE(const T& bi, const T& ci) {
        return bi + scalar * ci;
      }));
  }
};

struct nstream
{
  static constexpr size_t reads_per_item  = 3;
  static constexpr size_t writes_per_item = 1;

  template <typename T>
  static float64_t run(thrust::device_vector<T> a, thrust::device_vector<T> b, thrust::device_vector<T> c)
  {
    const T scalar = startScalar;
    return bench_transform(
      thrust::make_zip_iterator(a.begin(), b.begin(), c.begin()),
      thrust::make_zip_iterator(a.end(), b.end(), c.end()),
      a.begin(),
      thrust::make_zip_function(
        ::thrust::detail::proclaim_copyable_arguments([=] THRUST_DEVICE(const T& ai, const T& bi, const T& ci) {
          return ai + bi + scalar * ci;
        })));
  }
};

struct nstream_stable
{
  static constexpr size_t reads_per_item  = 3;
  static constexpr size_t writes_per_item = 1;

  template <typename T>
  static float64_t run(thrust::device_vector<T> a, thrust::device_vector<T> b, thrust::device_vector<T> c)
  {
    const T* a_start = thrust::raw_pointer_cast(a.data());
    const T* b_start = thrust::raw_pointer_cast(b.data());
    const T* c_start = thrust::raw_pointer_cast(c.data());
    const T scalar   = startScalar;
    return bench_transform(a.begin(), a.end(), a.begin(), [=] THRUST_DEVICE(const T& ai) {
      const auto i = &ai - a_start;
      return ai + b_start[i] + scalar * c_start[i];
    });
  }
};

template <typename Benchmark, class T>
void run_babelstream(benchmark::State& state, const std::size_t n)
{
  thrust::device_vector<T> a, b, c;
  try
  {
    a = thrust::device_vector<T>(n);
    b = thrust::device_vector<T>(n);
    c = thrust::device_vector<T>(n);
  }
  catch (const ::thrust::system::detail::bad_alloc& e)
  {
    (void) hipGetLastError();
    state.SkipWithError(("thrust::system::detail::bad_alloc: " + std::string(e.what())).c_str());
    return;
  }

  std::vector<double> gpu_times;
  for (auto _ : state)
  {
    thrust::fill(a.begin(), a.end(), startA);
    thrust::fill(b.begin(), b.end(), startB);
    thrust::fill(c.begin(), c.end(), startC);

    auto duration = Benchmark::template run<T>(a, b, c);
    state.SetIterationTime(duration);
    gpu_times.push_back(duration);
  }
  size_t transfers_per_item = Benchmark::reads_per_item + Benchmark::writes_per_item;
  state.SetBytesProcessed(state.iterations() * n * sizeof(T) * transfers_per_item);
  state.SetItemsProcessed(state.iterations() * n);

  const double gpu_cv         = bench_utils::StatisticsCV(gpu_times);
  state.counters["gpu_noise"] = gpu_cv;
}

#define CREATE_BABELSTREAM_BENCHMARK(T, Elements, Benchmark)                                             \
  benchmark::RegisterBenchmark(                                                                          \
    bench_utils::bench_naming::format_name(                                                              \
      "{algo:transform,subalgo:" + name + "." + #Benchmark + ",input_type:" #T + ",elements:" #Elements) \
      .c_str(),                                                                                          \
    run_babelstream<Benchmark, T>,                                                                       \
    Elements)

// clang-format off
// Different benchmarks use a different number of buffers. H200/B200 can fit 2^31 elements for all benchmarks and types.
// Upstream BabelStream uses 2^25. Allocation failure just skips the benchmark
#define BENCHMARK_BABELSTREAM_TYPE(type)              \
  CREATE_BABELSTREAM_BENCHMARK(type, 1 << 25, mul),   \
  CREATE_BABELSTREAM_BENCHMARK(type, 1 << 31, mul),   \
  CREATE_BABELSTREAM_BENCHMARK(type, 1 << 25, add),   \
  CREATE_BABELSTREAM_BENCHMARK(type, 1 << 31, add),   \
  CREATE_BABELSTREAM_BENCHMARK(type, 1 << 25, triad), \
  CREATE_BABELSTREAM_BENCHMARK(type, 1 << 31, triad), \
  CREATE_BABELSTREAM_BENCHMARK(type, 1 << 25, nstream), \
  CREATE_BABELSTREAM_BENCHMARK(type, 1 << 31, nstream), \
  CREATE_BABELSTREAM_BENCHMARK(type, 1 << 25, nstream_stable), \
  CREATE_BABELSTREAM_BENCHMARK(type, 1 << 31, nstream_stable)
// clang-format on

void add_benchmarks(const std::string& name, std::vector<benchmark::internal::Benchmark*>& benchmarks)
{
  std::vector<benchmark::internal::Benchmark*> bs = {
    BENCHMARK_BABELSTREAM_TYPE(int8_t),
    BENCHMARK_BABELSTREAM_TYPE(int16_t),
    BENCHMARK_BABELSTREAM_TYPE(float),
    BENCHMARK_BABELSTREAM_TYPE(double)
#if THRUST_BENCHMARKS_HAVE_INT128_SUPPORT
      ,
    BENCHMARK_BABELSTREAM_TYPE(int128_t)
#endif
  };

  benchmarks.insert(benchmarks.end(), bs.begin(), bs.end());
}
#undef CREATE_BABELSTREAM_BENCHMARK
#undef BENCHMARK_BABELSTREAM_TYPE
}; // namespace babelstream

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

  // Add benchmarks
  std::vector<benchmark::internal::Benchmark*> benchmarks;
  add_benchmarks<basic>("basic", benchmarks, seed_type);
  babelstream::add_benchmarks("babelstream", benchmarks);

  // Use manual timing
  for (auto& b : benchmarks)
  {
    b->UseManualTime();
    b->Unit(benchmark::kMicrosecond);
    b->MinTime(0.2); // in seconds
  }

  // Run benchmarks
  benchmark::RunSpecifiedBenchmarks(bench_utils::ChooseCustomReporter());

  // Finish
  benchmark::Shutdown();
  return 0;
}
